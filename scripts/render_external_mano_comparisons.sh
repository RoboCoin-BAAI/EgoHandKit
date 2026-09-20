#!/usr/bin/env bash
set -Eeuo pipefail

# Render the original front_output MANO predictions and place them beside the
# corresponding EgoHandKit overlays for visual comparison.
#
# Example:
#   ROOT=/data/export BACKEND=hamer CAMERAS=left,right FORCE=1 \
#     scripts/render_external_mano_comparisons.sh

ROOT="${ROOT:-/home/user/1/新数据测试}"
FRONT_OUTPUT="${FRONT_OUTPUT:-$ROOT/front_output}"
PROCESSED_ROOT="${PROCESSED_ROOT:-$ROOT/processed_data}"
EGO_VIDEO_ROOT="${EGO_VIDEO_ROOT:-$ROOT/generated_videos}"
EXTERNAL_OVERLAY_ROOT="${EXTERNAL_OVERLAY_ROOT:-$ROOT/external_mano_overlays}"
COMPARISON_ROOT="${COMPARISON_ROOT:-$ROOT/comparison_videos}"
MANO_RENDER_ENV="${MANO_RENDER_ENV:-egoall}"
BACKEND="${BACKEND:-hamer}"
CAMERAS="${CAMERAS:-left}"
COMPARE_HEIGHT="${COMPARE_HEIGHT:-720}"
FORCE="${FORCE:-0}"

if [[ "$FORCE" != "0" && "$FORCE" != "1" ]]; then
    echo "FORCE must be 0 or 1, got: $FORCE" >&2
    exit 2
fi
if [[ ! -d "$FRONT_OUTPUT/sequences" ]]; then
    echo "External parameter directory not found: $FRONT_OUTPUT/sequences" >&2
    exit 2
fi
for COMMAND in conda ffmpeg; do
    if ! command -v "$COMMAND" >/dev/null 2>&1; then
        echo "Required command not found: $COMMAND" >&2
        exit 2
    fi
done

mkdir -p "$EXTERNAL_OVERLAY_ROOT" "$COMPARISON_ROOT"
INDEX="$COMPARISON_ROOT/index.tsv"
printf 'session\tcamera\texternal_overlay\tegohandkit_overlay\tcomparison\tstatus\n' > "$INDEX"

while IFS= read -r -d '' PARAMETER; do
    CAMERA="$(basename "$PARAMETER" .npy)"
    if [[ "$CAMERAS" != "all" && ",$CAMERAS," != *",$CAMERA,"* ]]; then
        continue
    fi
    SESSION="$(basename "$(dirname "$PARAMETER")")"
    SOURCE_VIDEO="$PROCESSED_ROOT/$SESSION/videos/${CAMERA}_rectified.mp4"
    EXTERNAL_VIDEO="$EXTERNAL_OVERLAY_ROOT/$SESSION/${CAMERA}_overlay.mp4"
    EGO_VIDEO="$EGO_VIDEO_ROOT/${SESSION}_${CAMERA}_${BACKEND}_render.mp4"
    COMPARISON_VIDEO="$COMPARISON_ROOT/${SESSION}_${CAMERA}_${BACKEND}_comparison.mp4"

    if [[ ! -f "$SOURCE_VIDEO" ]]; then
        echo "[SKIP] missing source video: $SOURCE_VIDEO" >&2
        STATUS="missing_source_video"
    elif [[ ! -f "$EGO_VIDEO" ]]; then
        echo "[SKIP] missing EgoHandKit video: $EGO_VIDEO" >&2
        STATUS="missing_egohandkit_video"
    else
        if [[ "$FORCE" == "1" || ! -f "$EXTERNAL_VIDEO" ]]; then
            echo "[EXTERNAL] $SESSION / $CAMERA"
            conda run --no-capture-output -n "$MANO_RENDER_ENV" \
                python "$FRONT_OUTPUT/render_overlay.py" \
                --data-root "$PROCESSED_ROOT" \
                --output-root "$EXTERNAL_OVERLAY_ROOT" \
                --session "$SESSION" \
                --camera "$CAMERA"
        else
            echo "[CACHE] $EXTERNAL_VIDEO"
        fi

        if [[ ! -f "$EXTERNAL_VIDEO" ]]; then
            echo "[ERROR] external renderer did not produce: $EXTERNAL_VIDEO" >&2
            exit 1
        fi

        if [[ "$FORCE" == "1" || ! -f "$COMPARISON_VIDEO" ]]; then
            echo "[COMPARE] $SESSION / $CAMERA"
            ffmpeg -hide_banner -loglevel warning -y \
                -i "$EXTERNAL_VIDEO" \
                -i "$EGO_VIDEO" \
                -filter_complex \
                "[0:v]setpts=PTS-STARTPTS,scale=-2:${COMPARE_HEIGHT},setsar=1,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='EXTERNAL MANO':x=20:y=20:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.65[a];[1:v]setpts=PTS-STARTPTS,scale=-2:${COMPARE_HEIGHT},setsar=1,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='EGOHANDKIT ${BACKEND^^}':x=20:y=20:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.65[b];[a][b]hstack=inputs=2:shortest=1[v]" \
                -map "[v]" -an -shortest \
                -r 30 -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
                -movflags +faststart "$COMPARISON_VIDEO"
        else
            echo "[CACHE] $COMPARISON_VIDEO"
        fi
        STATUS="completed"
        echo "[VIDEO] $COMPARISON_VIDEO"
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$SESSION" "$CAMERA" "$EXTERNAL_VIDEO" "$EGO_VIDEO" "$COMPARISON_VIDEO" "$STATUS" >> "$INDEX"
done < <(find "$FRONT_OUTPUT/sequences" -mindepth 2 -maxdepth 2 -type f -name '*.npy' -print0 | sort -z)

echo
echo "External overlays: $EXTERNAL_OVERLAY_ROOT"
echo "Comparison videos: $COMPARISON_ROOT"
echo "Index: $INDEX"
