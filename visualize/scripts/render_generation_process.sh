#!/usr/bin/env bash
# Render generation process meshes via bproc-pubvis.
#
# Iterates over .ply files produced by vis_generation_process.py and renders
# each frame with bproc-pubvis. Output PNGs are saved alongside the .ply files
# (or in a separate render directory). Use render_generation_process.py to
# compose the PNGs into strips afterwards.
#
# Usage:
#   ./render_generation_process.sh <input_dir> [options]
#   ./render_generation_process.sh /path/to/generation_process --method diffusion
#
# Then compose strips:
#   python render_generation_process.py <input_dir> --labels

set -euo pipefail

# --- Configuration -----------------------------------------------------------
BPROC_PUBVIS="${BPROC_PUBVIS:-$(dirname "$(realpath "$0")")/../../../bproc-pubvis}"
RESOLUTION="${RESOLUTION:-512}"
SAMPLES="${SAMPLES:-64}"
NOISE_THRESHOLD="${NOISE_THRESHOLD:-0.05}"
CAM_LOCATION="${CAM_LOCATION:-1.5 0 1}"
CAM_OFFSET="${CAM_OFFSET:-0 0 0}"
# ShapeNet uses OpenGL convention (Y-up); Blender uses Z-up.
ROTATE="${ROTATE:-90 0 45}"
SHADE="${SHADE:-flat}"
# Keep object pose fixed across frames by default.
# bproc-pubvis per-file centering/scaling causes visible "jumps" in strips.
CENTER="${CENTER:-False}"
SCALE="${SCALE:-False}"
ORBIT_DEG="${ORBIT_DEG:-0}"
ORBIT_START_DEG="${ORBIT_START_DEG:-0}"

# Colors (linear RGB, matching paper figures / bproc-pubvis constants).
COLOR_INPUT="0.410603 0.101933 0.0683599"    # Color.PALE_RED  — salmon
COLOR_GT="0.410603 0.101933 0.0683599"       # Color.PALE_RED  — same
COLOR_GENERATED="0.165398 0.558341 0.416653" # Color.PALE_GREEN — teal

# --- Parse arguments ----------------------------------------------------------
METHOD="both"
VIEW="${VIEW:-}"
NAME_SUFFIX="${NAME_SUFFIX:-}"
TRANSPARENT="True"
NJOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)"
VERBOSE=""
QUIET=""

set_view_preset() {
    case "$1" in
        front)
            CAM_LOCATION="1.5 0 0"
            ROTATE="90 0 0"
            ;;
        back)
            CAM_LOCATION="1.5 0 0"
            ROTATE="90 0 180"
            ;;
        left)
            CAM_LOCATION="1.5 0 0"
            ROTATE="90 0 90"
            ;;
        right)
            CAM_LOCATION="1.5 0 0"
            ROTATE="90 0 -90"
            ;;
        top)
            CAM_LOCATION="0 0 1.8"
            ROTATE="90 0 180"
            ;;
        *)
            echo "Unknown view preset: $1 (expected: front|back|left|right|top)" >&2
            exit 1
            ;;
    esac
}

usage() {
    cat <<EOF
Usage: $(basename "$0") <input_dir> [options]

Options:
  --method diffusion|ar|both   Which method to render (default: both)
  --resolution N               Render resolution, square (default: 512)
  --samples N                  Cycles render samples (default: 64)
  --noise-threshold F          Cycles adaptive sampling threshold (default: 0.05)
  --view NAME                  Camera/view preset: front|back|left|right|top
  --name-suffix STR            Suffix added to output PNG names (default: view name)
  --cam-location X Y Z         Camera position (default: 1.5 0 1)
  --cam-offset X Y Z           Camera look-at offset (default: 0 0 0)
  --orbit-deg DEG              Orbit camera by DEG across timeline frames (default: 0, disabled)
  --orbit-start-deg DEG        Orbit start angle in degrees (default: 0)
  --rotate X Y Z               Object rotation in degrees (default: from --view preset)
  --opaque                     Opaque background (default: transparent)
  --center BOOL                Pass through to bproc-pubvis (default: False)
  --scale BOOL|FLOAT           Pass through to bproc-pubvis (default: False)
  --fast                       Fast preview: 256px, 32 samples, high noise threshold
  --parallel N                 Max parallel renders (default: nproc)
  --overwrite                  Re-render existing PNGs instead of skipping
  --verbose                    Verbose bproc-pubvis output
  --quiet                      Minimal wrapper output (errors only)
  -h, --help                   Show this help
EOF
    exit 0
}

OVERWRITE=""
INPUT_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)      METHOD="$2"; shift 2 ;;
        --resolution)  RESOLUTION="$2"; shift 2 ;;
        --samples)     SAMPLES="$2"; shift 2 ;;
        --noise-threshold) NOISE_THRESHOLD="$2"; shift 2 ;;
        --view)        VIEW="$2"; set_view_preset "$VIEW"; shift 2 ;;
        --name-suffix) NAME_SUFFIX="$2"; shift 2 ;;
        --cam-location) CAM_LOCATION="$2 $3 $4"; shift 4 ;;
        --cam-offset)  CAM_OFFSET="$2 $3 $4"; shift 4 ;;
        --orbit-deg)   ORBIT_DEG="$2"; shift 2 ;;
        --orbit-start-deg) ORBIT_START_DEG="$2"; shift 2 ;;
        --rotate)      ROTATE="$2 $3 $4"; shift 4 ;;
        --center)      CENTER="$2"; shift 2 ;;
        --scale)
            case "${2,,}" in
                true|yes|on|1) SCALE="True" ;;
                false|no|off|0) SCALE="False" ;;
                *) SCALE="$2" ;;
            esac
            shift 2
            ;;
        --opaque)      TRANSPARENT="False"; shift ;;
        --fast)        RESOLUTION=256; SAMPLES=32; NOISE_THRESHOLD=0.1; shift ;;
        --parallel)    if [[ "${2:-}" =~ ^[0-9]+$ ]]; then NJOBS="$2"; shift 2; else shift; fi ;;
        --overwrite)   OVERWRITE=1; shift ;;
        --verbose)     VERBOSE=1; shift ;;
        --quiet)       QUIET=1; shift ;;
        -h|--help)     usage ;;
        -*)            echo "Unknown option: $1" >&2; exit 1 ;;
        *)             INPUT_DIR="$1"; shift ;;
    esac
done

if [[ -n "$VERBOSE" && -n "$QUIET" ]]; then
    echo "Error: --verbose and --quiet are mutually exclusive." >&2
    exit 1
fi

log_info() {
    [[ -n "$QUIET" ]] && return 0
    echo "$@"
}

if [[ -z "$INPUT_DIR" ]]; then
    echo "Error: input_dir is required." >&2
    usage
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: $INPUT_DIR is not a directory." >&2
    exit 1
fi

if [[ ! -d "$BPROC_PUBVIS" ]]; then
    echo "Error: bproc-pubvis not found at $BPROC_PUBVIS" >&2
    echo "Set BPROC_PUBVIS=/path/to/bproc-pubvis" >&2
    exit 1
fi

if [[ -z "$NAME_SUFFIX" ]]; then
    NAME_SUFFIX="$VIEW"
fi

# --- Render function ----------------------------------------------------------
render_one() {
    local ply_path="$1"
    local png_path="$2"
    local color="$3"
    local extra_args="${4:-}"
    local orbit_idx="${5:--1}"
    local orbit_total="${6:-0}"
    local center_flag="--no-center"
    local cam_location_use="$CAM_LOCATION"
    case "${CENTER,,}" in
        1|true|yes|on) center_flag="--center" ;;
    esac

    if [[ "$ORBIT_DEG" != "0" && "$orbit_total" -gt 0 && "$orbit_idx" -ge 0 ]]; then
        local cam_x cam_y cam_z angle_deg orbit_xy orbit_x orbit_y
        read -r cam_x cam_y cam_z <<< "$CAM_LOCATION"
        angle_deg="$(
            awk -v s="$ORBIT_START_DEG" -v d="$ORBIT_DEG" -v i="$orbit_idx" -v n="$orbit_total" \
                'BEGIN{if(n<=1){printf "%.6f", s}else{printf "%.6f", s + d*(i/(n-1.0))}}'
        )"
        orbit_xy="$(
            awk -v x="$cam_x" -v y="$cam_y" -v a="$angle_deg" \
                'BEGIN{pi=atan2(0,-1); r=a*pi/180; ox=x*cos(r)-y*sin(r); oy=x*sin(r)+y*cos(r); printf "%.6f %.6f", ox, oy}'
        )"
        read -r orbit_x orbit_y <<< "$orbit_xy"
        cam_location_use="$orbit_x $orbit_y $cam_z"
    fi

    if [[ -z "$OVERWRITE" ]] && [[ -f "$png_path" ]]; then
        [[ "$NJOBS" -le 1 && -z "$QUIET" ]] && echo "  skip: $(basename "$png_path")"
        return 0
    fi

    [[ "$NJOBS" -le 1 && -z "$QUIET" ]] && echo "  render: $(basename "$ply_path") -> $(basename "$png_path")"
    local -a color_args cam_loc_args cam_off_args rot_args extra_arr
    read -r -a color_args <<< "$color"
    read -r -a cam_loc_args <<< "$cam_location_use"
    read -r -a cam_off_args <<< "$CAM_OFFSET"
    read -r -a rot_args <<< "$ROTATE"
    read -r -a extra_arr <<< "$extra_args"

    local -a cmd=(
        uv run blenderproc run main.py
        --data "$ply_path"
        --save "$png_path"
        --color "${color_args[@]}"
        --resolution "$RESOLUTION"
        --cam-location "${cam_loc_args[@]}"
        --cam-offset "${cam_off_args[@]}"
        --rotate "${rot_args[@]}"
        --shade "$SHADE"
        "$center_flag"
        --scale "$SCALE"
        --transparent "$TRANSPARENT"
        --samples "$SAMPLES"
        --noise-threshold "$NOISE_THRESHOLD"
    )
    if [[ ${#extra_arr[@]} -gt 0 ]]; then
        cmd+=("${extra_arr[@]}")
    fi
    if [[ -n "$VERBOSE" ]]; then
        cmd+=(--verbose)
    fi

    if [[ -n "$VERBOSE" ]]; then
        (cd "$BPROC_PUBVIS" && "${cmd[@]}")
        return
    fi

    local log_file
    log_file="$(mktemp /tmp/render_generation_process.XXXXXX.log)"
    if ! (cd "$BPROC_PUBVIS" && "${cmd[@]}") >"$log_file" 2>&1; then
        echo "  fail: $(basename "$png_path")" >&2
        echo "  renderer output (last 20 lines):" >&2
        tail -n 20 "$log_file" >&2 || true
        rm -f "$log_file"
        return 1
    fi
    rm -f "$log_file"
}

render_pointcloud() {
    local ply_path="$1"
    local png_path="$2"
    local color="$3"

    render_one "$ply_path" "$png_path" "$color" "--point-shape sphere"
}

# --- Find object directories --------------------------------------------------
mapfile -t OBJ_DIRS < <(
    find "$INPUT_DIR" -mindepth 2 -maxdepth 2 -type d | sort
)

# Fallback: input_dir itself might be an object directory.
if [[ ${#OBJ_DIRS[@]} -eq 0 ]]; then
    if [[ -d "$INPUT_DIR/diffusion" ]] || [[ -d "$INPUT_DIR/ar" ]]; then
        OBJ_DIRS=("$INPUT_DIR")
    fi
fi

if [[ ${#OBJ_DIRS[@]} -eq 0 ]]; then
    echo "No object directories found in $INPUT_DIR" >&2
    exit 1
fi

log_info "Found ${#OBJ_DIRS[@]} object(s) to render."

# --- Render loop --------------------------------------------------------------
METHODS=()
[[ "$METHOD" == "diffusion" || "$METHOD" == "both" ]] && METHODS+=("diffusion")
[[ "$METHOD" == "ar" || "$METHOD" == "both" ]] && METHODS+=("ar")

# Build a job list: each line is "ply_path png_path color extra_args".
JOBFILE="$(mktemp)"
trap 'rm -f "$JOBFILE"' EXIT

for obj_dir in "${OBJ_DIRS[@]}"; do
    for method in "${METHODS[@]}"; do
        method_dir="$obj_dir/$method"
        [[ -d "$method_dir" ]] || continue

        render_dir="$method_dir"
        mkdir -p "$render_dir"

        # Input point cloud.
        if [[ -f "$obj_dir/input.ply" ]]; then
            if [[ -n "$NAME_SUFFIX" ]]; then
                echo "$obj_dir/input.ply|$render_dir/input_${NAME_SUFFIX}.png|$COLOR_INPUT|--point-shape sphere" >> "$JOBFILE"
            else
                echo "$obj_dir/input.ply|$render_dir/input.png|$COLOR_INPUT|--point-shape sphere" >> "$JOBFILE"
            fi
        fi

        # Step/token meshes (sorted), with per-frame index for optional orbit.
        timeline_plys=()
        for ply in "$method_dir"/*.ply; do
            [[ -f "$ply" ]] || continue
            timeline_plys+=("$ply")
        done
        if [[ ${#timeline_plys[@]} -gt 0 ]]; then
            mapfile -t timeline_plys < <(printf '%s\n' "${timeline_plys[@]}" | sort)
            timeline_total=${#timeline_plys[@]}
            for i in "${!timeline_plys[@]}"; do
                ply="${timeline_plys[$i]}"
                stem="$(basename "$ply" .ply)"
                if [[ -n "$NAME_SUFFIX" ]]; then
                    echo "$ply|$render_dir/${stem}_${NAME_SUFFIX}.png|$COLOR_GENERATED||$i|$timeline_total" >> "$JOBFILE"
                else
                    echo "$ply|$render_dir/${stem}.png|$COLOR_GENERATED||$i|$timeline_total" >> "$JOBFILE"
                fi
            done
        fi

        # GT mesh.
        if [[ -f "$obj_dir/gt.ply" ]]; then
            if [[ -n "$NAME_SUFFIX" ]]; then
                echo "$obj_dir/gt.ply|$render_dir/gt_${NAME_SUFFIX}.png|$COLOR_GT|" >> "$JOBFILE"
            else
                echo "$obj_dir/gt.ply|$render_dir/gt.png|$COLOR_GT|" >> "$JOBFILE"
            fi
        fi
    done
done

TOTAL="$(wc -l < "$JOBFILE")"
log_info "Queued $TOTAL render job(s), parallel=$NJOBS."
log_info "Settings: view=${VIEW}, suffix=${NAME_SUFFIX}, ${RESOLUTION}px, ${SAMPLES} samples, noise_threshold=${NOISE_THRESHOLD}"
[[ -n "$OVERWRITE" ]] && log_info "Mode: overwrite existing PNGs"
log_info ""

run_job() {
    local line="$1"
    local orbit_idx orbit_total
    IFS='|' read -r ply_path png_path color extra_args orbit_idx orbit_total <<< "$line"
    render_one "$ply_path" "$png_path" "$color" "$extra_args" "${orbit_idx:--1}" "${orbit_total:-0}"
}
export -f run_job render_one render_pointcloud
export BPROC_PUBVIS RESOLUTION SAMPLES NOISE_THRESHOLD CAM_LOCATION CAM_OFFSET ROTATE SHADE CENTER SCALE ORBIT_DEG ORBIT_START_DEG TRANSPARENT VERBOSE QUIET OVERWRITE NJOBS
unset PARALLEL  # GNU parallel interprets $PARALLEL as its own CLI options

if [[ "$NJOBS" -gt 1 ]] && command -v parallel &>/dev/null; then
    if [[ -n "$QUIET" ]]; then
        parallel -j "$NJOBS" run_job :::: "$JOBFILE"
    else
        parallel -j "$NJOBS" --progress run_job :::: "$JOBFILE"
    fi
elif [[ "$NJOBS" -gt 1 ]]; then
    # Fallback to xargs if GNU parallel is not available.
    xargs -P "$NJOBS" -I{} bash -c 'run_job "$@"' _ {} < "$JOBFILE"
else
    COUNT=0
    while IFS= read -r line; do
        run_job "$line"
        COUNT=$((COUNT + 1))
        [[ -z "$QUIET" ]] && echo "  [$COUNT/$TOTAL]"
    done < "$JOBFILE"
fi

if [[ -z "$QUIET" ]]; then
    echo "" >&2
    echo "Rendering complete. Compose strips with:" >&2
    echo "  python render_generation_process.py $INPUT_DIR --labels" >&2
fi
