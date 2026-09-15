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
import json
import os
import shutil
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
from pipeline_artifacts import ArtifactStore
from observation_frontend.schema import canonical_sequence


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
    parser.add_argument('--frontend', choices=['legacy', 'observations', 'mint'], default='legacy',
                        help='Legacy YOLO/cleanup, ViTPose observations, or an external MINT prediction cache')
    parser.add_argument('--mint_predictions', type=str, default=None,
                        help='External MINT prediction cache (.npz); required with --frontend mint')
    parser.add_argument('--mint_mano_model_dir', type=str, default=None,
                        help='MANO model directory for decoding raw MINT hand[218] when *_joints_cam are absent')
    parser.add_argument('--mint_bbox_scale', type=float, default=1.0,
                        help='Square padding applied to projected MINT joints before HaMeR')
    parser.add_argument('--mint_presence_threshold', type=float, default=0.5,
                        help='Minimum MINT hand presence probability to emit an observation')
    parser.add_argument('--yolo_check', action='store_true', default=False,
                        help='Run full-image YOLO as a diagnostic comparison; never replaces MINT observations')
    parser.add_argument('--yolo_check_iou_threshold', type=float, default=0.1,
                        help='IoU threshold used by --yolo_check')
    parser.add_argument('--temporal_smoother', action='store_true', default=False,
                        help='Apply the MINT UKF/RTS MANO smoother after HMR inference')
    parser.add_argument('--endpoint_wrist_gate', action='store_true', default=False,
                        help='Reject only raw backend segment endpoints with an extreme wrist jump')
    parser.add_argument('--endpoint_wrist_max_deg', type=float, default=100.0,
                        help='Reject only raw backend segment endpoints whose wrist rotation jump to the adjacent frame exceeds this angle in degrees')
    parser.add_argument('--smoother_q', type=float, default=0.6,
                        help='MINT-style smoother process-noise scale')
    parser.add_argument('--smoother_r', type=float, default=0.6,
                        help='MINT-style smoother observation-noise scale')
    parser.add_argument('--smoother_beta', type=float, default=2.0,
                        help='MINT-style speed-adaptive noise scale')
    parser.add_argument('--depth_gate', action='store_true', default=False,
                        help='Reject MINT observations with invalid or implausible depth before HaMeR')
    parser.add_argument('--depth_dir', type=str, default=None,
                        help='Depth directory containing fast_foundation/depth_uint16_png')
    parser.add_argument('--depth_min_m', type=float, default=0.05,
                        help='Absolute minimum valid depth for the MINT depth gate')
    parser.add_argument('--depth_max_m', type=float, default=4.0,
                        help='Absolute maximum valid depth for the MINT depth gate')
    parser.add_argument('--depth_max_ratio', type=float, default=2.5,
                        help='Maximum current/reference depth ratio')
    parser.add_argument('--depth_min_ratio', type=float, default=0.4,
                        help='Minimum current/reference depth ratio')
    parser.add_argument('--depth_history', type=int, default=10,
                        help='Number of recent valid depths used as the relative reference')
    parser.add_argument('--depth_max_bad_frames', type=int, default=2,
                        help='Consecutive bad-frame budget recorded by the gate; any bad frame starts a new fragment')
    parser.add_argument('--motion_gate', action='store_true', default=False,
                        help='Reject implausible image-space jumps before HaMeR')
    parser.add_argument('--motion_center_threshold', type=float, default=0.25,
                        help='Maximum bbox-center jump as a fraction of image diagonal')
    parser.add_argument('--motion_size_ratio', type=float, default=2.0,
                        help='Maximum bbox area ratio for a motion-gate vote')
    parser.add_argument('--motion_iou_threshold', type=float, default=0.1,
                        help='Minimum bbox IoU for a motion-gate vote')
    parser.add_argument('--motion_joint_threshold', type=float, default=0.25,
                        help='Maximum projected-joint jump as a fraction of image diagonal')
    parser.add_argument('--motion_min_votes', type=int, default=2,
                        help='Number of failed motion tests required to reject a sample')
    parser.add_argument('--motion_reacquire_frames', type=int, default=2,
                        help='Lookahead frames used to classify a persistent jump')
    parser.add_argument('--observation_all_person', action='store_true', default=False,
                        help='Observation experiment: run ViTPose on every detected person instead of the highest-score wearer')
    parser.add_argument('--observation_no_consolidation', action='store_true', default=False,
                        help='Observation ablation: keep overlapping same-frame proposals instead of removing duplicates')
    parser.add_argument('--motion_prediction_weight', type=float, default=0.0,
                        help='Optional robust constant-velocity prior for observations association')
    parser.add_argument('--switch_penalty', type=float, default=0.0,
                        help='Optional penalty for high-cost continued track assignments')
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
    if args.frontend == 'mint' and (args.use_vitpose or args.no_clean_bbox):
        parser.error('--frontend mint consumes external predictions; omit --use_vitpose and --no_clean_bbox')
    if args.frontend == 'mint' and not args.mint_predictions:
        parser.error('--frontend mint requires --mint_predictions /path/to/mint_predictions.npz')
    if args.frontend != 'mint' and args.mint_predictions:
        parser.error('--mint_predictions is only valid with --frontend mint')
    if args.mint_bbox_scale <= 0:
        parser.error('--mint_bbox_scale must be positive')
    if not 0 <= args.mint_presence_threshold <= 1:
        parser.error('--mint_presence_threshold must lie in [0,1]')
    if not 0 <= args.yolo_check_iou_threshold <= 1:
        parser.error('--yolo_check_iou_threshold must lie in [0,1]')
    if args.endpoint_wrist_max_deg <= 0:
        parser.error('--endpoint_wrist_max_deg must be positive')
    if args.depth_gate and not args.depth_dir:
        parser.error('--depth_gate requires --depth_dir')
    if args.motion_center_threshold <= 0 or args.motion_size_ratio < 1:
        parser.error('--motion_center_threshold must be positive and --motion_size_ratio must be >= 1')
    if not 0 <= args.motion_iou_threshold <= 1 or args.motion_joint_threshold <= 0:
        parser.error('--motion_iou_threshold must lie in [0,1] and joint threshold must be positive')
    if not 1 <= args.motion_min_votes <= 4 or args.motion_reacquire_frames < 1:
        parser.error('--motion_min_votes must be 1..4 and reacquire frames must be positive')
    if args.smoother_q <= 0 or args.smoother_r <= 0 or args.smoother_beta < 0:
        parser.error('--smoother_q/r must be positive and beta must be non-negative')
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

    output_name = f'{video_name}_{args.frontend}'
    out_dir = Path(args.output_root) / output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = ArtifactStore(out_dir)
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
    artifacts.write_manifest({"schema_version": "egohand.run.v1", "sequence": {"name": video_name,
        "input": str(input_path.resolve()), "fps": float(fps)},
        "frontend": {"name": args.frontend}, "backend": {"name": args.backend},
        "force_detect": bool(args.force_detect), "pipeline": []})

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
    depth_report = None
    motion_report = None

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
        from observation_frontend.physical_hand_temporal_association import (
            DEFAULT_CONFIG as DEFAULT_ASSOCIATION_CONFIG,
        )
        from dataclasses import replace

        association_config = replace(
            DEFAULT_ASSOCIATION_CONFIG,
            motion_prediction_weight=args.motion_prediction_weight,
            high_cost_switch_penalty=args.switch_penalty,
        )

        cleaned_data = run_observation_frontend(
            img_paths, out_dir, repo_root, device, fps=fps, force=args.force_detect,
            association_config=association_config,
            all_person=args.observation_all_person,
            enable_consolidation=not args.observation_no_consolidation,
        )
    elif args.frontend == 'mint':
        from observation_frontend.mint_adapter import (
            build_mint_observations,
            load_mint_observation_cache,
            load_mint_predictions,
            save_mint_observation_cache,
        )
        from observation_frontend.depth_gate import apply_depth_gate
        from observation_frontend.motion_gate import apply_motion_gate
        mint_cache = Path(args.mint_predictions)
        if not mint_cache.is_file():
            raise FileNotFoundError(f"MINT prediction cache does not exist: {mint_cache}")
        saved_mint_cache = out_dir / 'mint_predictions.npz'
        if mint_cache.resolve() != saved_mint_cache.resolve() and (args.force_detect or not saved_mint_cache.exists()):
            shutil.copy2(mint_cache, saved_mint_cache)
            print(f"MINT prediction cache copied to {saved_mint_cache}")
        observation_cache = out_dir / 'mint_observations.pkl'
        mint_config = {
            'bbox_scale': float(args.mint_bbox_scale),
            'presence_threshold': float(args.mint_presence_threshold),
            'mano_model_dir': str(Path(args.mint_mano_model_dir).resolve()) if args.mint_mano_model_dir else None,
            'projection': 'camera_frame_opencv_to_original_pixels_v1',
            'depth_gate': bool(args.depth_gate),
            'depth_dir': str(Path(args.depth_dir).resolve()) if args.depth_dir else None,
            'depth_min_m': float(args.depth_min_m),
            'depth_max_m': float(args.depth_max_m),
            'depth_max_ratio': float(args.depth_max_ratio),
            'depth_min_ratio': float(args.depth_min_ratio),
            'depth_history': int(args.depth_history),
            'depth_max_bad_frames': int(args.depth_max_bad_frames),
            'motion_gate': bool(args.motion_gate),
            'motion_center_threshold': float(args.motion_center_threshold),
            'motion_size_ratio': float(args.motion_size_ratio),
            'motion_iou_threshold': float(args.motion_iou_threshold),
            'motion_joint_threshold': float(args.motion_joint_threshold),
            'motion_min_votes': int(args.motion_min_votes),
            'motion_reacquire_frames': int(args.motion_reacquire_frames),
        }
        if observation_cache.exists() and not args.force_detect:
            cleaned_data = load_mint_observation_cache(
                observation_cache, image_paths=img_paths, mint_path=mint_cache, config=mint_config)
            print(f"Loading MINT observation cache: {observation_cache}")
        else:
            mint_predictions = load_mint_predictions(mint_cache)
            cleaned_data = build_mint_observations(
                mint_predictions, img_paths,
                bbox_scale=args.mint_bbox_scale,
                presence_threshold=args.mint_presence_threshold,
                mano_model_dir=args.mint_mano_model_dir,
            )
            if args.depth_gate:
                cleaned_data, depth_report = apply_depth_gate(
                    cleaned_data, img_paths, args.depth_dir,
                    min_depth_m=args.depth_min_m,
                    max_depth_m=args.depth_max_m,
                    max_ratio=args.depth_max_ratio,
                    min_ratio=args.depth_min_ratio,
                    history_size=args.depth_history,
                    max_bad_frames=args.depth_max_bad_frames,
                )
                (out_dir / 'depth_gate.json').write_text(
                    json.dumps(depth_report, indent=2) + '\n')
                print(f"Depth gate rejected {depth_report['rejected_observations']} observations")
            if args.motion_gate:
                cleaned_data, motion_report = apply_motion_gate(
                    cleaned_data, img_paths,
                    center_jump_threshold=args.motion_center_threshold,
                    size_ratio_threshold=args.motion_size_ratio,
                    iou_threshold=args.motion_iou_threshold,
                    joint_jump_threshold=args.motion_joint_threshold,
                    min_bad_votes=args.motion_min_votes,
                    reacquire_frames=args.motion_reacquire_frames,
                )
                (out_dir / 'motion_gate.json').write_text(
                    json.dumps(motion_report, indent=2) + '\n')
                print(f"Motion gate rejected {motion_report['counts']['rejected']} observations")
            save_mint_observation_cache(
                observation_cache, cleaned_data,
                image_paths=img_paths, mint_path=mint_cache, config=mint_config)
            print(f"MINT observations cached to {observation_cache}")
        if args.yolo_check:
            from observation_frontend.mint_adapter import compare_mint_yolo_observations
            print(f"\nRunning YOLO diagnostic check with {args.yolo_model}")
            if yolo_detector is None:
                yolo_detector = YOLO(args.yolo_model)
                yolo_detector.to(device)
            yolo_frames = extract_raw_bboxes(img_paths, yolo_detector, vis_dir=None, fps=fps)
            yolo_report = compare_mint_yolo_observations(
                cleaned_data, yolo_frames,
                iou_threshold=args.yolo_check_iou_threshold,
            )
            yolo_stage = artifacts.stage_dir("30_yolo_check")
            artifacts.write_json(yolo_stage / 'report.json', {**yolo_report, "decision_effect": "none",
                "schema_version": "egohand.yolo_check.v1"})
            print(f"YOLO diagnostic saved to {yolo_stage / 'report.json'}")
            print(f"YOLO check counts: {yolo_report['counts']}")
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

    # Apply optional frontend-independent gates for non-MINT adapters.  The
    # MINT branch applies these while building its observation cache.
    if args.frontend != 'mint' and (args.depth_gate or args.motion_gate):
        from observation_frontend.depth_gate import apply_depth_gate
        from observation_frontend.motion_gate import apply_motion_gate
        if args.depth_gate:
            cleaned_data, depth_report = apply_depth_gate(
                cleaned_data, img_paths, args.depth_dir, min_depth_m=args.depth_min_m,
                max_depth_m=args.depth_max_m, max_ratio=args.depth_max_ratio,
                min_ratio=args.depth_min_ratio, history_size=args.depth_history,
                max_bad_frames=args.depth_max_bad_frames)
        if args.motion_gate:
            cleaned_data, motion_report = apply_motion_gate(
                cleaned_data, img_paths, center_jump_threshold=args.motion_center_threshold,
                size_ratio_threshold=args.motion_size_ratio, iou_threshold=args.motion_iou_threshold,
                joint_jump_threshold=args.motion_joint_threshold, min_bad_votes=args.motion_min_votes,
                reacquire_frames=args.motion_reacquire_frames)

    # Persist the stable frontend boundary artifact.  Runtime legacy fields are
    # accepted by the adapter, but downstream consumers can inspect this file
    # without knowing which frontend produced it.
    canonical = canonical_sequence(cleaned_data, sequence_name=video_name, fps=fps, frontend=args.frontend)
    frontend_stage = artifacts.stage_dir("00_frontend")
    artifacts.write_pickle(frontend_stage / "observations.pkl", canonical)
    artifacts.write_json(frontend_stage / "summary.json", {"schema_version": "egohand.observations.v1",
        "frame_count": len(canonical["frames"]), "frontend": args.frontend,
        "loaded_from_cache": bool(not args.force_detect)})
    if args.depth_gate:
        depth_stage = artifacts.stage_dir("10_depth_gate")
        artifacts.write_pickle(depth_stage / "observations.pkl", canonical)
        artifacts.write_json(depth_stage / "report.json", depth_report or {"schema_version": "egohand.depth_gate.v1", "enabled": True})
    if args.motion_gate:
        motion_stage = artifacts.stage_dir("20_motion_gate")
        artifacts.write_pickle(motion_stage / "observations.pkl", canonical)
        artifacts.write_json(motion_stage / "report.json", motion_report or {"schema_version": "egohand.motion_gate.v1", "enabled": True})
    manifest = {"schema_version": "egohand.run.v1", "sequence": {"name": video_name,
        "input": str(input_path.resolve()), "frame_count": len(img_paths), "fps": float(fps)},
        "frontend": {"name": args.frontend}, "backend": {"name": args.backend},
        "force_detect": bool(args.force_detect), "pipeline": [{"name": "frontend", "enabled": True,
            "artifact": "stages/00_frontend/observations.pkl"}]}
    artifacts.write_manifest(manifest)

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

    final_dir = artifacts.final
    final_frames = []
    for frame in canonical["frames"]:
        output = results.get(frame["img_path"], {})
        hands = []
        for index, mano in enumerate(output.get("mano", [])):
            hands.append({"track_id": int(output.get("tracked_ids", [])[index]),
                "fragment_id": 0, "handedness": "right" if int(output.get("tracked_ids", [])[index]) else "left",
                "backend_handedness": "right" if int(output.get("tracked_ids", [])[index]) else "left",
                "mano": mano, "cam_trans": output.get("cam_trans", [])[index],
                "keypoints_2d": output.get("extra_data", [])[index], "meta": output.get("backend_meta", [])[index]})
        final_frames.append({"frame_idx": frame["frame_idx"], "img_path": frame["img_path"],
            "timestamp_ns": frame.get("timestamp_ns"), "hands": hands})
    final_payload = {"schema_version": "egohand.results.v1", "sequence": canonical["sequence"],
        "frontend": args.frontend, "backend": args.backend,
        "coordinate_system": {"image": "original_pixels", "camera": "opencv_x_right_y_down_z_forward"},
        "frames": final_frames, "summary": {"frame_count": len(final_frames),
            "hand_instance_count": sum(len(frame["hands"]) for frame in final_frames),
            "hands_by_side": {"left": sum(1 for frame in final_frames for hand in frame["hands"] if hand["handedness"] == "left"),
                               "right": sum(1 for frame in final_frames for hand in frame["hands"] if hand["handedness"] == "right")}}}
    artifacts.write_pickle(final_dir / "results.pkl", final_payload)
    artifacts.write_json(final_dir / "summary.json", {"schema_version": "egohand.summary.v1", **final_payload["summary"],
        "frontend": {"name": args.frontend}, "backend": args.backend})
    manifest["pipeline"].extend([
        {"name": "depth_gate", "enabled": bool(args.depth_gate), "artifact": "stages/10_depth_gate/observations.pkl" if args.depth_gate else None},
        {"name": "motion_gate", "enabled": bool(args.motion_gate), "artifact": "stages/20_motion_gate/observations.pkl" if args.motion_gate else None},
        {"name": "yolo_check", "enabled": bool(args.yolo_check), "artifact": "stages/30_yolo_check/report.json" if args.yolo_check else None},
        {"name": "backend_raw", "enabled": True, "artifact": "stages/40_backend_raw/outputs.pkl"},
        {"name": "endpoint_wrist_gate", "enabled": bool(args.endpoint_wrist_gate), "artifact": "stages/50_endpoint_wrist_gate/outputs.pkl" if args.endpoint_wrist_gate else None},
        {"name": "temporal_smoother", "enabled": bool(args.temporal_smoother), "artifact": "stages/60_smoother/outputs.pkl" if args.temporal_smoother else None},
    ])
    manifest["final_artifacts"] = {"results": "final/results.pkl", "summary": "final/summary.json",
        "render": "final/render.mp4" if args.render else None}
    artifacts.write_manifest(manifest)

    with open(res_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {res_path}")

    if args.omega_world:
        # Pass 4 uses a second large model.  Release the HMR model first so
        # Omega's chunk can fit on GPUs with limited VRAM (for example 12 GB).
        omega_backend_name = backend_bundle.backend_name
        omega_model_cfg = backend_bundle.model_cfg
        del model
        backend_bundle.model = None
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
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
            backend_name=omega_backend_name,
            backend_model_cfg=omega_model_cfg,
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
