#!/usr/bin/env bash
set -Eeuo pipefail

# Run the same left-camera input through legacy and observations frontends,
# then create synchronized camera-space and Omega world-space comparisons.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/home/user/ego_data/测试数据}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/user/ego_data/结果对比/egohandkit_left_batch}"
GPU="${GPU:-0}"
IMG_FOCAL="${IMG_FOCAL:-600}"
BATCH_SIZE="${BATCH_SIZE:-2}"
OMEGA_CHUNK_SIZE="${OMEGA_CHUNK_SIZE:-16}"
OMEGA_OVERLAP="${OMEGA_OVERLAP:-8}"
OMEGA_IMAGE_RESOLUTION="${OMEGA_IMAGE_RESOLUTION:-512}"
FORCE="${FORCE:-0}"

command -v ffmpeg >/dev/null || { echo "ffmpeg is required" >&2; exit 1; }
PYTHON="${PYTHON:-python}"
mkdir -p "$OUTPUT_ROOT"

shopt -s nullglob
videos=("$DATA_ROOT"/*/videos/left.mp4)
if ((${#videos[@]} == 0)); then
  echo "No left-camera videos found under: $DATA_ROOT" >&2
  exit 1
fi

for video in "${videos[@]}"; do
  session="$(basename "$(dirname "$(dirname "$video")")")"
  session_root="$OUTPUT_ROOT/$session"
  before_root="$session_root/before"
  after_root="$session_root/after"
  mkdir -p "$session_root"

  # run.py derives this name for videos under a videos/ directory.
  sequence="${session}_left"
  before_video="$before_root/$sequence/render_hawor.mp4"
  after_video="$after_root/${sequence}_observations/render_hawor.mp4"
  before_world="$before_root/$sequence/omega_world_grid_hawor.mp4"
  after_world="$after_root/${sequence}_observations/omega_world_grid_hawor.mp4"

  echo
  echo "===== $session / left ====="
  if [[ "$FORCE" == 1 || ! -f "$before_video" || ! -f "$before_world" ]]; then
    "$PYTHON" -u "$ROOT/run.py" \
      --input "$video" --backend hawor --frontend legacy --gpu "$GPU" \
      --batch_size "$BATCH_SIZE" --img_focal "$IMG_FOCAL" --omega_world \
      --omega_chunk_size "$OMEGA_CHUNK_SIZE" --omega_overlap "$OMEGA_OVERLAP" \
      --omega_image_resolution "$OMEGA_IMAGE_RESOLUTION" \
      --output_root "$before_root" 2>&1 | tee "$session_root/before.log"
  else
    echo "Reuse: $before_video"
  fi

  if [[ "$FORCE" == 1 || ! -f "$after_video" || ! -f "$after_world" ]]; then
    "$PYTHON" -u "$ROOT/run.py" \
      --input "$video" --backend hawor --frontend observations --gpu "$GPU" \
      --batch_size "$BATCH_SIZE" --img_focal "$IMG_FOCAL" --omega_world \
      --omega_chunk_size "$OMEGA_CHUNK_SIZE" --omega_overlap "$OMEGA_OVERLAP" \
      --omega_image_resolution "$OMEGA_IMAGE_RESOLUTION" \
      --output_root "$after_root" 2>&1 | tee "$session_root/after.log"
  else
    echo "Reuse: $after_video"
  fi

  if [[ ! -f "$before_video" || ! -f "$after_video" || ! -f "$before_world" || ! -f "$after_world" ]]; then
    echo "Missing expected output for $session; inspect logs under $session_root" >&2
    exit 1
  fi

  ffmpeg -hide_banner -loglevel error -y \
    -i "$before_video" -i "$after_video" \
    -filter_complex "[0:v]setpts=PTS-STARTPTS,scale=-2:544[left];[1:v]setpts=PTS-STARTPTS,scale=-2:544[right];[left][right]hstack=inputs=2[v]" \
    -map "[v]" -an -c:v libx264 -pix_fmt yuv420p -crf 18 -preset medium \
    "$session_root/camera_before_after.mp4"

  ffmpeg -hide_banner -loglevel error -y \
    -i "$before_world" -i "$after_world" \
    -filter_complex "[0:v]setpts=PTS-STARTPTS,scale=-2:544[left];[1:v]setpts=PTS-STARTPTS,scale=-2:544[right];[left][right]hstack=inputs=2[v]" \
    -map "[v]" -an -c:v libx264 -pix_fmt yuv420p -crf 18 -preset medium \
    "$session_root/world_before_after.mp4"

  printf '%s\n' "session=$session" "input=$video" "focal=$IMG_FOCAL" \
    "camera_comparison=$session_root/camera_before_after.mp4" \
    "world_comparison=$session_root/world_before_after.mp4" > "$session_root/README.txt"
  echo "Camera comparison: $session_root/camera_before_after.mp4"
  echo "World comparison:  $session_root/world_before_after.mp4"
done

echo
echo "All left-camera comparisons are under: $OUTPUT_ROOT"
