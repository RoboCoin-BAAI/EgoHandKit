#!/usr/bin/env bash

set -Eeo pipefail


# ============================================================
# Fixed config
# ============================================================

# Conda environments
MINT_ENV="mint-inference"
ACE_ENV="ace-ego-hand"
EGOHANDKIT_ENV="egohandkit"

# Project roots
EGOHANDKIT_ROOT="/home/user/EgoHandKit"
MINT_ROOT="/home/user/wuji-ego-mint"
ACE_ROOT="/home/user/ACE-Ego-Hand"

# ------------------------------------------------------------
# Exporters
#
# EgoHandKit contains the source copies.
# If frontend repo does not contain export_egohandkit_cache.py,
# copy the corresponding exporter into its scripts/ directory.
# ------------------------------------------------------------

LOCAL_MINT_EXPORTER="${EGOHANDKIT_ROOT}/scripts/export_egohandkit_cache_mint.py"
LOCAL_ACE_EXPORTER="${EGOHANDKIT_ROOT}/scripts/export_egohandkit_cache_ace.py"

MINT_EXPORTER="${MINT_ROOT}/scripts/export_egohandkit_cache.py"
ACE_EXPORTER="${ACE_ROOT}/scripts/export_egohandkit_cache.py"

# ------------------------------------------------------------
# MINT inference
# ------------------------------------------------------------

MINT_CHECKPOINT="${MINT_ROOT}/checkpoints/model.safetensors"
MINT_TARGET_FPS=30
MINT_WINDOW=32

# ------------------------------------------------------------
# ACE inference
# ------------------------------------------------------------

ACE_OPT="${ACE_ROOT}/options/ace_ego_hand_k.yml"
ACE_CHECKPOINT="${ACE_ROOT}/checkpoints/ace_ego_hand_k.pt"
ACE_ENCODE_W=832

# Temporary fixed camera intrinsics:
# fx,fy,cx,cy
ACE_INTRINSICS="448.31524557986006,448.31524557986006,906.4268105579597,547.236132483085"

# ------------------------------------------------------------
# EgoHandKit
# ------------------------------------------------------------

BACKEND="hamer"
GPU=0


# ============================================================
# Helpers
# ============================================================

die() {
    echo "[ERROR] $*" >&2
    exit 1
}


usage() {
    cat <<EOF
Usage:
  $0 <mint|ace> <video> <depth_dir> <output_root>

MINT:
  $0 mint <video.mp4> <depth_dir> <output_root>

ACE:
  $0 ace <video.mp4> <depth_dir> <output_root>

Example:

  $0 mint \
    /home/user/data/short/left_rectified.mp4 \
    /home/user/data/short/depth \
    /home/user/data/short/pipeline_output

  $0 ace \
    /home/user/data/short/left_rectified.mp4 \
    /home/user/data/short/depth \
    /home/user/data/short/pipeline_output
EOF
}


ensure_exporter() {
    local frontend="$1"
    local source_exporter
    local target_exporter

    case "$frontend" in

        mint)
            source_exporter="$LOCAL_MINT_EXPORTER"
            target_exporter="$MINT_EXPORTER"
            ;;

        ace)
            source_exporter="$LOCAL_ACE_EXPORTER"
            target_exporter="$ACE_EXPORTER"
            ;;

        *)
            die "unknown frontend for exporter: $frontend"
            ;;
    esac

    if [[ -f "$target_exporter" ]] && cmp -s "$source_exporter" "$target_exporter"; then
        echo "[exporter] up to date: $target_exporter"
        return
    fi

    echo "[exporter] synchronizing from EgoHandKit"

    [[ -f "$source_exporter" ]] \
        || die "source exporter not found: $source_exporter"

    mkdir -p "$(dirname "$target_exporter")"

    cp "$source_exporter" "$target_exporter"

    [[ -s "$target_exporter" ]] \
        || die "failed to install exporter: $target_exporter"

    echo "[exporter] installed: $target_exporter"
}


# ============================================================
# Args
# ============================================================

[[ $# -eq 4 ]] || {
    usage
    exit 1
}

FRONTEND="$1"
VIDEO_INPUT="$2"
DEPTH_INPUT="$3"
OUTPUT_INPUT="$4"

[[ "$FRONTEND" == "mint" || "$FRONTEND" == "ace" ]] \
    || die "frontend must be 'mint' or 'ace'"

[[ -f "$VIDEO_INPUT" ]] \
    || die "video not found: $VIDEO_INPUT"

[[ -d "$DEPTH_INPUT" ]] \
    || die "depth directory not found: $DEPTH_INPUT"

[[ -d "$EGOHANDKIT_ROOT" ]] \
    || die "EgoHandKit root not found: $EGOHANDKIT_ROOT"

[[ -f "${EGOHANDKIT_ROOT}/run.py" ]] \
    || die "EgoHandKit run.py not found: ${EGOHANDKIT_ROOT}/run.py"

command -v conda >/dev/null 2>&1 \
    || die "conda not found"


# ============================================================
# Resolve paths
# ============================================================

VIDEO="$(realpath "$VIDEO_INPUT")"
DEPTH_DIR="$(realpath "$DEPTH_INPUT")"

mkdir -p "$OUTPUT_INPUT"
OUTPUT_ROOT="$(realpath "$OUTPUT_INPUT")"

VIDEO_NAME="$(basename "$VIDEO")"
VIDEO_STEM="${VIDEO_NAME%.*}"


# ============================================================
# Output paths
# ============================================================

# Completely isolate MINT and ACE.
EGO_OUTPUT_ROOT="${OUTPUT_ROOT}/${FRONTEND}"

# Raw frontend inference outputs.
INFERENCE_ROOT="${EGO_OUTPUT_ROOT}/inference"

# Canonical EgoHandKit observations.
CACHE_ROOT="${EGO_OUTPUT_ROOT}/canonical_inputs"

mkdir -p "$INFERENCE_ROOT"
mkdir -p "$CACHE_ROOT"

OBSERVATIONS="${CACHE_ROOT}/${VIDEO_STEM}_${FRONTEND}.observations.pkl"


echo "============================================================"
echo "Frontend    : $FRONTEND"
echo "Video       : $VIDEO"
echo "Depth       : $DEPTH_DIR"
echo "Inference   : $INFERENCE_ROOT"
echo "Canonical   : $OBSERVATIONS"
echo "Output root : $EGO_OUTPUT_ROOT"
echo "Backend     : $BACKEND"
echo "GPU         : $GPU"
echo "============================================================"


# ============================================================
# Step 0: ensure exporter exists
# ============================================================

echo "[0/3] Check exporter"

ensure_exporter "$FRONTEND"


# ============================================================
# Step 1: frontend inference
# ============================================================

if [[ "$FRONTEND" == "mint" ]]; then

    # --------------------------------------------------------
    # MINT
    # --------------------------------------------------------

    [[ -d "$MINT_ROOT" ]] \
        || die "MINT root not found: $MINT_ROOT"

    [[ -f "$MINT_CHECKPOINT" ]] \
        || die "MINT checkpoint not found: $MINT_CHECKPOINT"

    [[ -f "$MINT_EXPORTER" ]] \
        || die "MINT exporter not found: $MINT_EXPORTER"

    echo "[1/3] MINT inference"

    (
        cd "$MINT_ROOT"

        conda run \
            --no-capture-output \
            -n "$MINT_ENV" \
            python -m mint infer \
                --input "$VIDEO" \
                --checkpoint "$MINT_CHECKPOINT" \
                --output "$INFERENCE_ROOT" \
                --target-fps "$MINT_TARGET_FPS" \
                --window "$MINT_WINDOW"
    )

    RAW_RESULT="${INFERENCE_ROOT}/prediction.npz"

    [[ -s "$RAW_RESULT" ]] \
        || die "MINT prediction not generated: $RAW_RESULT"


else

    # --------------------------------------------------------
    # ACE
    # --------------------------------------------------------

    [[ -d "$ACE_ROOT" ]] \
        || die "ACE root not found: $ACE_ROOT"

    [[ -f "${ACE_ROOT}/infer_video.py" ]] \
        || die "ACE infer_video.py not found: ${ACE_ROOT}/infer_video.py"

    [[ -f "$ACE_OPT" ]] \
        || die "ACE option file not found: $ACE_OPT"

    [[ -f "$ACE_CHECKPOINT" ]] \
        || die "ACE checkpoint not found: $ACE_CHECKPOINT"

    [[ -f "$ACE_EXPORTER" ]] \
        || die "ACE exporter not found: $ACE_EXPORTER"

    echo "[1/3] ACE inference"

    (
        cd "$ACE_ROOT"

        conda run \
            --no-capture-output \
            -n "$ACE_ENV" \
            python infer_video.py \
                --video "$VIDEO" \
                --intrinsics "$ACE_INTRINSICS" \
                --opt "$ACE_OPT" \
                --ckpt "$ACE_CHECKPOINT" \
                --out "$INFERENCE_ROOT" \
                --encode_w "$ACE_ENCODE_W"
    )

    RAW_RESULT="${INFERENCE_ROOT}/${VIDEO_STEM}.pkl"

    [[ -s "$RAW_RESULT" ]] \
        || die "ACE prediction not generated: $RAW_RESULT"

fi


echo "Prediction  : $RAW_RESULT"


# ============================================================
# Step 2: frontend result -> canonical observations
# ============================================================

echo "[2/3] ${FRONTEND^^} -> canonical"

if [[ "$FRONTEND" == "mint" ]]; then

    conda run \
        --no-capture-output \
        -n "$EGOHANDKIT_ENV" \
        python "$MINT_EXPORTER" \
            --source "$VIDEO" \
            --raw "$RAW_RESULT" \
            --out "$OBSERVATIONS"

else

    conda run \
        --no-capture-output \
        -n "$EGOHANDKIT_ENV" \
        python "$ACE_EXPORTER" \
            --source "$VIDEO" \
            --raw "$RAW_RESULT" \
            --out "$OBSERVATIONS"

fi

[[ -s "$OBSERVATIONS" ]] \
    || die "canonical observation not generated: $OBSERVATIONS"


# ============================================================
# Step 3: canonical -> EgoHandKit HaMeR chain
# ============================================================

echo "[3/3] EgoHandKit"

(
    cd "$EGOHANDKIT_ROOT"

    conda run \
        --no-capture-output \
        -n "$EGOHANDKIT_ENV" \
        python run.py \
            --input "$VIDEO" \
            --frontend canonical \
            --observations "$OBSERVATIONS" \
            --backend "$BACKEND" \
            --depth_gate \
            --depth_dir "$DEPTH_DIR" \
            --depth_max_m 1.0 \
            --mint_depth_sensor_anchor \
            --motion_gate \
            --mint_3d_consistency_gate \
            --endpoint_wrist_gate \
            --endpoint_wrist_max_deg 100 \
            --temporal_smoother \
            --hand_tracking_parquet \
            --force_detect \
            --gpu "$GPU" \
            --output_root "$EGO_OUTPUT_ROOT"
)


# ============================================================
# Done
# ============================================================

echo "============================================================"
echo "DONE"
echo "Frontend   : $FRONTEND"
echo "Prediction : $RAW_RESULT"
echo "Canonical  : $OBSERVATIONS"
echo "Output     : $EGO_OUTPUT_ROOT"
echo "============================================================"
