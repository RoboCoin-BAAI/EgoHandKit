#!/usr/bin/env bash

set -Eeo pipefail

# ============================================================
# Fixed config
# ============================================================

CONDA_ENV="egohandkit"
EGOHANDKIT_ROOT="/home/user/EgoHandKit"
MINT_ROOT="/home/user/wuji-ego-mint"
ACE_ROOT="/home/user/ACE-Ego-Hand"
MINT_EXPORTER="$MINT_ROOT/scripts/export_egohandkit_cache.py"
ACE_EXPORTER="$ACE_ROOT/scripts/export_egohandkit_cache.py"

BACKEND="hamer"
GPU=0


die() {
    echo "[ERROR] $*" >&2
    exit 1
}

usage() {
    cat <<EOF
Usage:
  $0 <mint|ace> <video> <prediction> <depth_dir> <output_root>

MINT:
  $0 mint <video.mp4> <prediction.npz> <depth_dir> <output_root>

ACE:
  $0 ace <video.mp4> <prediction.pkl> <depth_dir> <output_root>

Example:
  $0 mint \
    /home/user/1/long/left_rectified_600.mp4 \
    /home/user/1/long/mint_result/prediction.npz \
    /home/user/1/long/session_xxx/depth \
    /home/user/1/long/canonical_hamer_test
EOF
}


# ============================================================
# Args
# ============================================================

[[ $# -eq 5 ]] || {
    usage
    exit 1
}

FRONTEND="$1"
VIDEO_INPUT="$2"
RAW_INPUT="$3"
DEPTH_INPUT="$4"
OUTPUT_INPUT="$5"

[[ "$FRONTEND" == "mint" || "$FRONTEND" == "ace" ]] \
    || die "frontend must be 'mint' or 'ace'"

[[ -f "$VIDEO_INPUT" ]] \
    || die "video not found: $VIDEO_INPUT"

[[ -f "$RAW_INPUT" ]] \
    || die "prediction not found: $RAW_INPUT"

[[ -d "$DEPTH_INPUT" ]] \
    || die "depth directory not found: $DEPTH_INPUT"

[[ -f "${EGOHANDKIT_ROOT}/run.py" ]] \
    || die "EgoHandKit run.py not found: ${EGOHANDKIT_ROOT}/run.py"

VIDEO="$(realpath "$VIDEO_INPUT")"
RAW_RESULT="$(realpath "$RAW_INPUT")"
DEPTH_DIR="$(realpath "$DEPTH_INPUT")"

mkdir -p "$OUTPUT_INPUT"
OUTPUT_ROOT="$(realpath "$OUTPUT_INPUT")"


# ============================================================
# Activate environment
# ============================================================

command -v conda >/dev/null 2>&1 \
    || die "conda not found"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"


# ============================================================
# Output paths
# ============================================================

VIDEO_NAME="$(basename "$VIDEO")"
VIDEO_STEM="${VIDEO_NAME%.*}"

# Keep MINT and ACE isolated under the requested output root.
EGO_OUTPUT_ROOT="${OUTPUT_ROOT}/${FRONTEND}"
CACHE_ROOT="${EGO_OUTPUT_ROOT}/canonical_inputs"

mkdir -p "$CACHE_ROOT"

OBSERVATIONS="${CACHE_ROOT}/${VIDEO_STEM}_${FRONTEND}.observations.pkl"


echo "============================================================"
echo "Frontend    : $FRONTEND"
echo "Video       : $VIDEO"
echo "Prediction  : $RAW_RESULT"
echo "Depth       : $DEPTH_DIR"
echo "Canonical   : $OBSERVATIONS"
echo "Output root : $EGO_OUTPUT_ROOT"
echo "Backend     : $BACKEND"
echo "GPU         : $GPU"
echo "============================================================"


# ============================================================
# Step 1: existing frontend result -> canonical observations
# ============================================================

if [[ "$FRONTEND" == "mint" ]]; then

    [[ -f "$MINT_EXPORTER" ]] \
        || die "MINT exporter not found: $MINT_EXPORTER"

    echo "[1/2] MINT -> canonical"

    python "$MINT_EXPORTER" \
        --source "$VIDEO" \
        --raw "$RAW_RESULT" \
        --out "$OBSERVATIONS"

else

    [[ -f "$ACE_EXPORTER" ]] \
        || die "ACE exporter not found: $ACE_EXPORTER"

    echo "[1/2] ACE -> canonical"

    python "$ACE_EXPORTER" \
        --source "$VIDEO" \
        --raw "$RAW_RESULT" \
        --out "$OBSERVATIONS"

fi

[[ -s "$OBSERVATIONS" ]] \
    || die "canonical observation not generated: $OBSERVATIONS"


# ============================================================
# Step 2: canonical -> EgoHandKit HaMeR test chain
# ============================================================

echo "[2/2] EgoHandKit"

cd "$EGOHANDKIT_ROOT"

python run.py \
    --input "$VIDEO" \
    --frontend canonical \
    --observations "$OBSERVATIONS" \
    --backend hamer \
    --depth_gate \
    --depth_dir "$DEPTH_DIR" \
    --depth_max_m 1.0 \
    --motion_gate \
    --yolo_check \
    --endpoint_wrist_gate \
    --endpoint_wrist_max_deg 100 \
    --temporal_smoother \
    --force_detect \
    --gpu 0 \
    --output_root "$EGO_OUTPUT_ROOT"


echo "============================================================"
echo "DONE"
echo "Canonical : $OBSERVATIONS"
echo "Output    : $EGO_OUTPUT_ROOT"
echo "============================================================"
