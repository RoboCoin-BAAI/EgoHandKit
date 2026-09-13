"""EgoHandKit: hand mesh recovery and world-space MANO reconstruction.

This is the main entry point for a 4-pass video/image-sequence pipeline:

Pass 1: detect left/right hand bounding boxes with YOLO, optionally merged with
        Detectron2 + ViTPose.
Pass 2: clean hand bbox tracks over time, unless disabled by --no_clean_bbox.
Pass 3: infer MANO parameters with one HMR backend. The default backend is
        HaWoR, with HaMeR, HTM, and WiLoR also supported.
Pass 4: recover per-frame Omega cameras and derive world-space MANO results.

Primary outputs are written under test_data/hand_proc/{sequence_name}/:

- {sequence}_{backend}.pkl: camera-space MANO inference results.
- omega_camera.npz: Omega camera intrinsics/extrinsics and preprocessing meta.
- {sequence}_{backend}_omega_world.pkl: world-space MANO-derived results.
- render_{backend}.mp4: original-camera mesh overlay video.
- omega_world_grid_{backend}.mp4: 2x2 fixed-view world visualization.

Hand overlays are the default. Use --omega_world to opt into world recovery.
"""

from pathlib import Path
import argparse
import gc
import os
import sys

# ---------------------------------------------------------------------------
# GPU selection — must happen before importing torch / CUDA-consuming libraries
# so that the process can only see the requested physical GPU.
# ---------------------------------------------------------------------------
_GPU_IDX = 0


def _parse_gpu_early():
    global _GPU_IDX
    _GPU_IDX = 0
    for i, arg in enumerate(sys.argv):
        if arg == '--gpu' and i + 1 < len(sys.argv):
            try:
                _GPU_IDX = int(sys.argv[i + 1])
            except ValueError:
                pass
        elif arg.startswith('--gpu='):
            try:
                _GPU_IDX = int(arg.split('=', 1)[1])
            except ValueError:
                pass
    if os.environ.get('CUDA_VISIBLE_DEVICES') is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(_GPU_IDX)
    os.environ['EGL_DEVICE_ID'] = str(_GPU_IDX)


_parse_gpu_early()

import torch
import cv2
import numpy as np
import pickle
from tqdm import tqdm
from ultralytics import YOLO

from hmr_backends.utils.renderer import Renderer
from hmr_backends.models import load_backend
from mesh_recovery import run_mesh_recovery

from bbox_utils import (
    create_video_from_images,
    convert_crop_coords_to_orig_img,
    extract_raw_bboxes,
    extract_raw_bboxes_vitpose,
    load_or_build_cleaned_bboxes,
    merge_detections,
    clean_bbox_sequences,
)
from input_naming import sequence_name_for_input


VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


def extract_video_frames(video_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if native_fps <= 0:
        cap.release()
        raise ValueError(f"Cannot read FPS from video: {video_path}")
    existing = sorted(output_dir.glob('*.jpg'))
    if len(existing) > 0:
        print(f"Frames already extracted in {output_dir} ({len(existing)} frames), skipping.")
        cap.release()
        return native_fps
    print(f"\nExtracting frames from {video_path}")
    print(f"  Native FPS: {native_fps:.1f}, Total frames: {total_frames}")
    saved_count = 0
    pbar = tqdm(total=total_frames, desc="Extracting frames")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = output_dir / f'{saved_count:06d}.jpg'
        cv2.imwrite(str(out_path), frame)
        saved_count += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    print(f"  Extracted {saved_count} frames to {output_dir}")
    return native_fps


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='EgoHandKit: 4-pass hand mesh recovery and world-space MANO reconstruction'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input video file or image folder')
    parser.add_argument('--sequence_name', type=str, default=None,
                        help='Output/cache sequence name. Defaults to session_camera for dataset videos')
    parser.add_argument('--backend', type=str, default='hawor', choices=['hamer', 'htm', 'wilor', 'hawor'],
                        help='Backend model to use (hamer, htm, wilor, or hawor)')
    parser.add_argument('--frontend', choices=['legacy', 'observations'], default='legacy',
                        help='Legacy YOLO/cleanup or all-person ViTPose observation selection (offline)')
    parser.add_argument('--output_root', type=str, default='test_data/hand_proc',
                        help='Root directory for outputs. Default: test_data/hand_proc')
    parser.add_argument('--fps', type=int, default=15,
                        help='FPS for output visualization videos (only used for image folder input; video input auto-uses native FPS)')
    parser.add_argument('--render', dest='render', action='store_true', default=True,
                        help='If set, render mesh overlay results')
    parser.add_argument('--force_detect', action='store_true', default=False,
                        help='Force re-run Pass 1/2 even if cached results exist')
    parser.add_argument('--no_clean_bbox', action='store_true', default=False,
                        help='Skip Pass 2 bbox temporal cleaning and feed raw Pass 1 detections directly to Pass 3')
    parser.add_argument('--use_vitpose', action='store_true', default=False,
                        help='Also run Detectron2+ViTPose detector and merge with YOLO (per-hand best confidence)')
    parser.add_argument('--batch_size', type=int, default=48, help='Batch size for inference')
    parser.add_argument('--rescale_factor', type=float, default=None,
                        help='Factor for padding the bbox (default: 2.0 for hamer/htm/wilor; user-overridable, e.g. 1.3; hawor uses its own crop pipeline)')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'],
                        help='List of file extensions to consider')
    parser.add_argument('--yolo_model', type=str, default='./_DATA/hand_det/detector.pt',
                        help='Path to YOLO hand detector model')
    parser.add_argument('--img_focal', type=float, default=None,
                        help='Image focal length for HaWoR backend; if omitted follow HaWoR logic: est_focal.txt then default 600')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Physical CUDA GPU index to use when CUDA is available (0-based). This is applied before importing torch by setting CUDA_VISIBLE_DEVICES.')
    omega_mode = parser.add_mutually_exclusive_group()
    omega_mode.add_argument('--omega_world', action='store_true', default=False,
                           help='Opt into Omega camera recovery and world-space MANO derivation')
    omega_mode.add_argument('--no_omega_world', dest='omega_world', action='store_false',
                           help='Disable Omega (already the default)')
    parser.add_argument('--omega_checkpoint', type=str, default='./_DATA/vggt_omega/vggt_omega_1b_512.pt',
                        help='Path to Omega checkpoint')
    parser.add_argument('--omega_image_resolution', type=int, default=512,
                        help='Omega preprocessing image resolution')
    parser.add_argument('--omega_chunk_size', default='auto',
                        help="Omega chunk size: 'auto' or a positive integer")
    parser.add_argument('--omega_overlap', type=int, default=8,
                        help='Number of overlapping frames for Omega chunk alignment')
    parser.add_argument('--omega_force', action='store_true', default=False,
                        help='Force re-run Omega camera recovery even if omega_camera.npz exists')
    parser.add_argument('--no_omega_world_vis', dest='omega_world_vis', action='store_false', default=True,
                        help='Disable default Omega world fixed-view 2x2 visualization video')
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.frontend == 'observations' and (args.use_vitpose or args.no_clean_bbox):
        parser.error('--frontend observations replaces legacy detection/cleanup; omit --use_vitpose and --no_clean_bbox')
    if args.fps <= 0:
        parser.error('--fps must be positive')

    input_path = Path(args.input)
    if input_path.is_file() and input_path.suffix.lower() in VIDEO_EXTS:
        video_name = sequence_name_for_input(input_path, args.sequence_name)
        img_folder = Path('test_data/images') / video_name
        native_fps = extract_video_frames(input_path, img_folder)
        fps = native_fps
    elif input_path.is_dir():
        video_name = sequence_name_for_input(input_path, args.sequence_name)
        img_folder = input_path
        fps = args.fps
    else:
        raise ValueError(f"--input must be a video file ({', '.join(VIDEO_EXTS)}) or image folder: {input_path}")

    output_name = video_name if args.frontend == 'legacy' else f'{video_name}_observations'
    out_dir = Path(args.output_root) / output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / f'{video_name}_{args.backend}.pkl'

    if torch.cuda.is_available():
        # CUDA_VISIBLE_DEVICES has already remapped the selected physical GPU to runtime cuda:0.
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print(f"\n{'='*60}")
    print(f"Input:   {img_folder}")
    print(f"Output:  {out_dir}")
    print(f"Backend: {args.backend}")
    print(f"FPS:     {fps}")
    print(f"GPU:     {args.gpu} (physical)")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"Runtime device: {device}")
    print(f"Result:  {res_path}")
    print(f"{'='*60}")

    if args.rescale_factor is None:
        # Unified default across hamer / htm / wilor. Users can override, e.g. --rescale_factor 1.3
        args.rescale_factor = 2.0 if args.backend in ('hamer', 'htm', 'wilor') else 1.3

    img_paths = sorted([img for end in args.file_type for img in img_folder.glob(end)])
    print(f"Found {len(img_paths)} images in {img_folder}")
    if len(img_paths) == 0:
        raise ValueError(f"No images found in {img_folder} matching {args.file_type}")

    repo_root = os.path.dirname(os.path.abspath(__file__))
    pass1_cache = out_dir / 'pass1_raw.pkl'
    pass2_cache = out_dir / 'pass2_cleaned.pkl'
    need_detect = args.frontend == 'legacy' and (args.force_detect or not pass1_cache.exists())

    yolo_detector = None
    body_detector = None
    vitpose = None

    if need_detect:
        print(f"\nLoading YOLO hand detector: {args.yolo_model}")
        yolo_detector = YOLO(args.yolo_model)
        yolo_detector.to(device)
        if args.use_vitpose:
            print("\nLoading Detectron2 ViTDet body detector...")
            from hmr_backends.utils.utils_detectron2 import DefaultPredictor_Lazy
            from detectron2.config import LazyConfig
            import hmr_backends as hmr_backends_pkg
            cfg_path = Path(hmr_backends_pkg.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
            detectron2_cfg = LazyConfig.load(str(cfg_path))
            detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
            for i in range(3):
                detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
            body_detector = DefaultPredictor_Lazy(detectron2_cfg, device=device)
            print("Loading ViTPose wholebody model...")
            from vitpose_model import ViTPoseModel
            vitpose = ViTPoseModel(repo_root, device)

    if args.frontend == 'observations':
        from observation_frontend.adapter import run_observation_frontend

        cleaned_data = run_observation_frontend(
            img_paths, out_dir, repo_root, device, fps=fps, force=args.force_detect,
        )
    elif pass1_cache.exists() and not args.force_detect:
        print(f"\nLoading cached Pass 1 results from {pass1_cache}")
        with open(pass1_cache, 'rb') as f:
            raw_data = pickle.load(f)
        print(f"  Loaded {len(raw_data)} frames")
    else:
        vis_dir = str(out_dir) if args.render else None
        yolo_data = extract_raw_bboxes(img_paths, yolo_detector, vis_dir=vis_dir, fps=fps)
        if args.use_vitpose:
            vitpose_data = extract_raw_bboxes_vitpose(img_paths, body_detector, vitpose, vis_dir=vis_dir, fps=fps)
            raw_data = merge_detections(yolo_data, vitpose_data)
        else:
            raw_data = yolo_data
        with open(pass1_cache, 'wb') as f:
            pickle.dump(raw_data, f)
        print(f"Pass 1 cached to {pass1_cache}")

    if args.frontend == 'legacy':
        pass2_vis_dir = str(out_dir) if args.render else None
        cleaned_data = load_or_build_cleaned_bboxes(
            raw_data, pass2_cache, force_detect=args.force_detect,
            no_clean_bbox=args.no_clean_bbox, clean_fn=clean_bbox_sequences,
            vis_dir=pass2_vis_dir, fps=fps,
        )

    # Frontend models are no longer needed while reconstructing hand crops.
    del yolo_detector, body_detector, vitpose
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    backend_bundle = load_backend(args.backend, repo_root, args)
    model = backend_bundle.model.to(device)
    model.eval()
    renderer = Renderer(backend_bundle.render_cfg, faces=model.mano.faces)

    # Pass 3 - MESH RECOVERY
    results = run_mesh_recovery(cleaned_data, backend_bundle, renderer, args, out_dir=out_dir, fps=fps, device=device)

    with open(res_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {res_path}")

    if args.omega_world:
        print("\n" + "=" * 80)
        print("Pass 4 - OMEGA WORLD: camera recovery and MANO world-space derivation")
        print("=" * 80)
        from hmr_backends.omega.runner import run_omega_camera_recovery
        from hmr_backends.omega.world import derive_omega_world_results
        from hmr_backends.omega.visualization import render_omega_world_grid_video

        omega_camera = run_omega_camera_recovery(
            image_paths=[str(p) for p in img_paths],
            out_dir=out_dir,
            checkpoint_path=args.omega_checkpoint,
            image_resolution=args.omega_image_resolution,
            chunk_size=args.omega_chunk_size,
            overlap=args.omega_overlap,
            force=args.omega_force,
            device=device,
        )
        omega_world_path = out_dir / f'{video_name}_{args.backend}_omega_world.pkl'
        omega_world_results = derive_omega_world_results(
            camera_space_results=results,
            omega_camera=omega_camera,
            backend_name=backend_bundle.backend_name,
            backend_model_cfg=backend_bundle.model_cfg,
            source_pkl_path=res_path,
            omega_cache_path=out_dir / 'omega_camera.npz',
            output_path=omega_world_path,
            device=device,
        )
        print(f"Omega world-space results saved to {omega_world_path}")
        if args.render and args.omega_world_vis:
            world_vis = render_omega_world_grid_video(
                omega_world_results=omega_world_results,
                image_paths=[str(p) for p in img_paths],
                out_dir=out_dir,
                backend_name=backend_bundle.backend_name,
                renderer=renderer,
                fps=fps,
            )
            print(f"Omega world fixed-view visualization saved to {world_vis['video_path']}")
        elif not args.omega_world_vis:
            print("Omega world fixed-view visualization skipped by --no_omega_world_vis")
    else:
        print("\nPass 4 - OMEGA WORLD: disabled (use --omega_world to enable)")


if __name__ == '__main__':
    main()
