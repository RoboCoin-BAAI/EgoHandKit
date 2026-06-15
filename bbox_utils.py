"""
Shared bbox detection, cleaning, and utility functions.
Used by both HaMeR and WiLoR 3-pass pipelines.
"""

from pathlib import Path
import os
import pickle
import cv2
import numpy as np
from tqdm import tqdm
import imageio

def compute_bbox_from_keypoints(keypoints, img_shape, scale_factor=1.0, max_size_ratio=0.4):
    """Compute tight bbox from keypoints. Scaling is applied later when creating HaMeR input patches."""
    valid = keypoints[:, 2] > 0.5
    if valid.sum() < 10:
        return None
    
    valid_kp = keypoints[valid, :2]
    x_min, y_min = valid_kp.min(axis=0)
    x_max, y_max = valid_kp.max(axis=0)
    
    width = x_max - x_min
    height = y_max - y_min
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    
    width *= scale_factor
    height *= scale_factor
    
    img_h, img_w = img_shape[:2]
    max_width = img_w * max_size_ratio
    max_height = img_h * max_size_ratio
    
    width = min(width, max_width)
    height = min(height, max_height)
    size = max(width, height)
    
    x_min = cx - size / 2
    y_min = cy - size / 2
    x_max = cx + size / 2
    y_max = cy + size / 2
    
    return np.array([x_min, y_min, x_max, y_max])


def compute_iou(bbox1, bbox2):
    """Compute IoU between two bboxes."""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = bbox1_area + bbox2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def compute_containment_ratio(bbox1, bbox2):
    """
    Compute how much bbox1 is contained within bbox2.
    Returns the ratio of bbox1's area that overlaps with bbox2.
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    
    return inter_area / bbox1_area if bbox1_area > 0 else 0


def enlarge_bbox(bbox, scale=1.2, img_shape=None):
    """Expand bbox to a square scaled by `scale`, optionally clipped to image bounds."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0

    size = max(w, h) * scale

    new_x1 = cx - size / 2.0
    new_y1 = cy - size / 2.0
    new_x2 = cx + size / 2.0
    new_y2 = cy + size / 2.0

    if img_shape is not None:
        H, W = img_shape[:2]
        new_x1 = max(0, min(W - 1, new_x1))
        new_y1 = max(0, min(H - 1, new_y1))
        new_x2 = max(0, min(W - 1, new_x2))
        new_y2 = max(0, min(H - 1, new_y2))

    return np.array([new_x1, new_y1, new_x2, new_y2], dtype=np.float64 if not hasattr(bbox, 'dtype') else bbox.dtype)


def create_video_from_images(image_folder, output_video_path, fps=30):
    """Create MP4 video from images in a folder using imageio (same as src_cam video)."""
    image_folder = Path(image_folder)
    if not image_folder.exists():
        print(f"Warning: {image_folder} does not exist, skipping video creation")
        return
    
    # Get all jpg images sorted by filename
    image_files = sorted(list(image_folder.glob('*.jpg')))
    if len(image_files) == 0:
        print(f"Warning: No images found in {image_folder}, skipping video creation")
        return
    
    print(f"Creating video: {output_video_path}")
    
    # Load all images
    frames = []
    for img_file in tqdm(image_files, desc=f"  Loading frames"):
        img = imageio.imread(str(img_file))
        if img is not None:
            frames.append(img)
    
    if len(frames) == 0:
        print(f"Warning: Could not load any images from {image_folder}")
        return
    
    # Write video using imageio (same method as src_cam video generation)
    imageio.mimwrite(str(output_video_path), frames, fps=fps)
    print(f"  Video saved: {output_video_path} ({len(frames)} frames @ {fps} fps)")


def load_or_build_cleaned_bboxes(
    raw_data,
    pass2_cache,
    *,
    force_detect=False,
    no_clean_bbox=False,
    clean_fn=None,
    vis_dir=None,
    fps=15,
):
    """Resolve Pass 2 bbox data, optionally bypassing cleaning entirely."""
    pass2_cache = Path(pass2_cache)
    if no_clean_bbox:
        print("\nPass 2 - BBOX CLEANING: skipped by --no_clean_bbox")
        return raw_data

    if pass2_cache.exists() and not force_detect:
        print(f"\nLoading cached Pass 2 results from {pass2_cache}")
        with open(pass2_cache, 'rb') as f:
            cleaned_data = pickle.load(f)
        print(f"  Loaded {len(cleaned_data)} frames")
        return cleaned_data

    if clean_fn is None:
        clean_fn = clean_bbox_sequences
    cleaned_data = clean_fn(raw_data, vis_dir=vis_dir, fps=fps)
    with open(pass2_cache, 'wb') as f:
        pickle.dump(cleaned_data, f)
    print(f"Pass 2 cached to {pass2_cache}")
    return cleaned_data


# ============================================================================
# PASS 1: Extract all raw bboxes using YOLO hand detector
# ============================================================================

def extract_raw_bboxes(img_paths, detector, vis_dir=None, fps=15):
    """
    Pass 1: Extract raw hand bboxes using YOLO hand detector (like WiLoR).
    YOLO directly detects hand bboxes with left/right classification.
    No ViTPose needed - much simpler and more reliable!
    
    Args:
        img_paths: List of image paths
        detector: YOLO hand detector model
        vis_dir: Optional directory for visualizations
    
    Returns:
        raw_data: List of dicts for each frame with bbox data (no keypoints from YOLO)
    """
    print("\n" + "="*80)
    print("PASS 1: Extracting raw hand bboxes (YOLO hand detector)")
    print("="*80)
    
    if vis_dir is not None:
        pass1_dir = os.path.join(vis_dir, 'pass1_raw_bboxes')
        os.makedirs(pass1_dir, exist_ok=True)
    
    raw_data = []
    
    for frame_idx, img_path in enumerate(tqdm(sorted(img_paths), desc="Pass 1: Extracting hand bboxes")):
        img_path = str(img_path)
        img_cv2 = cv2.imread(img_path)
        
        frame_data = {
            'frame_idx': frame_idx,
            'img_path': img_path,
            'left_bbox': None,
            'right_bbox': None,
            'left_raw_bbox': None,
            'right_raw_bbox': None,
            'left_keypoints': None,
            'right_keypoints': None,
            'left_conf': 0.0,
            'right_conf': 0.0,
            'left_track_id': None,
            'right_track_id': None,
        }

        # YOLO hand detection with ByteTrack tracking
        # Class 0 = left hand, Class 1 = right hand
        results = detector.track(img_cv2, conf=0.3, persist=True, verbose=False)
        detections = results[0] if results else None

        left_detections = []
        right_detections = []

        img_shape = img_cv2.shape
        if detections is not None and detections.boxes is not None and len(detections.boxes) > 0:
            boxes_xyxy = detections.boxes.xyxy.cpu().numpy()
            confs = detections.boxes.conf.cpu().numpy()
            classes = detections.boxes.cls.cpu().numpy().astype(int)
            track_ids = detections.boxes.id
            if track_ids is not None:
                track_ids = track_ids.cpu().numpy().astype(int)
            else:
                track_ids = [None] * len(boxes_xyxy)

            for i in range(len(boxes_xyxy)):
                raw_bbox = boxes_xyxy[i].copy()
                bbox = enlarge_bbox(boxes_xyxy[i], scale=1.2, img_shape=img_shape)
                conf = float(confs[i])
                cls = classes[i]
                tid = int(track_ids[i]) if track_ids[i] is not None else None

                if cls == 0:
                    left_detections.append({'bbox': bbox, 'raw_bbox': raw_bbox, 'conf': conf, 'track_id': tid})
                elif cls == 1:
                    right_detections.append({'bbox': bbox, 'raw_bbox': raw_bbox, 'conf': conf, 'track_id': tid})

        if len(left_detections) > 0:
            best_left = max(left_detections, key=lambda x: x['conf'])
            frame_data['left_bbox'] = best_left['bbox']
            frame_data['left_raw_bbox'] = best_left['raw_bbox']
            frame_data['left_conf'] = best_left['conf']
            frame_data['left_track_id'] = best_left['track_id']

        if len(right_detections) > 0:
            best_right = max(right_detections, key=lambda x: x['conf'])
            frame_data['right_bbox'] = best_right['bbox']
            frame_data['right_raw_bbox'] = best_right['raw_bbox']
            frame_data['right_conf'] = best_right['conf']
            frame_data['right_track_id'] = best_right['track_id']
        
        # Visualize raw bboxes
        if vis_dir is not None:
            img_vis = img_cv2.copy()
            if frame_data['left_bbox'] is not None:
                x1, y1, x2, y2 = frame_data['left_bbox'].astype(int)
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for left
                cv2.putText(img_vis, f"L T={frame_data['left_track_id']} ({frame_data['left_conf']:.2f})",
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if frame_data['right_bbox'] is not None:
                x1, y1, x2, y2 = frame_data['right_bbox'].astype(int)
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for right
                cv2.putText(img_vis, f"R T={frame_data['right_track_id']} ({frame_data['right_conf']:.2f})",
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            img_fn = os.path.splitext(os.path.basename(img_path))[0]
            cv2.imwrite(os.path.join(pass1_dir, f'{img_fn}_raw.jpg'), img_vis)
        
        raw_data.append(frame_data)
    
    # Print statistics with missing frame details
    left_count = sum(1 for f in raw_data if f['left_bbox'] is not None)
    right_count = sum(1 for f in raw_data if f['right_bbox'] is not None)
    missing_left = [i for i, f in enumerate(raw_data) if f['left_bbox'] is None]
    missing_right = [i for i, f in enumerate(raw_data) if f['right_bbox'] is None]
    
    print(f"\nExtracted bboxes for {len(raw_data)} frames:")
    print(f"  Left hand detected in {left_count} frames ({100*left_count/len(raw_data):.1f}%)")
    print(f"  Right hand detected in {right_count} frames ({100*right_count/len(raw_data):.1f}%)")
    
    if len(missing_left) > 0:
        print(f"  Missing left hand in {len(missing_left)} frames")
        if len(missing_left) <= 20:
            print(f"    Frame indices: {missing_left}")
        else:
            print(f"    First 10: {missing_left[:10]}")
            print(f"    Last 10: {missing_left[-10:]}")
    
    if len(missing_right) > 0:
        print(f"  Missing right hand in {len(missing_right)} frames")
        if len(missing_right) <= 20:
            print(f"    Frame indices: {missing_right}")
        else:
            print(f"    First 10: {missing_right[:10]}")
            print(f"    Last 10: {missing_right[-10:]}")
    
    # Create video from Pass 1 visualizations
    if vis_dir is not None:
        pass1_dir = os.path.join(vis_dir, 'pass1_raw_bboxes')
        video_path = os.path.join(vis_dir, 'pass1_raw_bboxes.mp4')
        create_video_from_images(pass1_dir, video_path, fps=fps)

    return raw_data


def extract_raw_bboxes_vitpose(img_paths, body_detector, vitpose_model, vis_dir=None,
                                person_score_threshold=0.5, keypoint_conf_threshold=0.5,
                                min_valid_keypoints=3, fps=15):
    """
    Pass 1 (ViTPose variant): Detectron2 body detection -> ViTPose 133-keypoint -> hand bboxes.
    Picks the single highest-confidence person per frame (at most 1 left + 1 right hand).

    Args:
        img_paths: List of image paths
        body_detector: Detectron2 DefaultPredictor_Lazy
        vitpose_model: ViTPoseModel instance
        vis_dir: Optional directory for visualizations
    Returns:
        raw_data: Same format as extract_raw_bboxes
    """
    print("\n" + "="*80)
    print("PASS 1 (ViTPose): Extracting hand bboxes via Detectron2 + ViTPose")
    print("="*80)

    if vis_dir is not None:
        pass1_vit_dir = os.path.join(vis_dir, 'pass1_raw_bboxes_vitpose')
        os.makedirs(pass1_vit_dir, exist_ok=True)

    raw_data = []

    for frame_idx, img_path in enumerate(tqdm(sorted(img_paths), desc="Pass 1 (ViTPose): Extracting hand bboxes")):
        img_path = str(img_path)
        img_cv2 = cv2.imread(img_path)
        img_rgb = img_cv2[:, :, ::-1]

        frame_data = {
            'frame_idx': frame_idx,
            'img_path': img_path,
            'left_bbox': None,
            'right_bbox': None,
            'left_raw_bbox': None,
            'right_raw_bbox': None,
            'left_keypoints': None,
            'right_keypoints': None,
            'left_conf': 0.0,
            'right_conf': 0.0,
        }

        # Detectron2 person detection
        det_out = body_detector(img_cv2)
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > person_score_threshold)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        if len(pred_bboxes) == 0:
            raw_data.append(frame_data)
            continue

        # Pick best person (highest Detectron2 score)
        best_idx = np.argmax(pred_scores)
        best_bbox = pred_bboxes[best_idx:best_idx+1]
        best_score = pred_scores[best_idx:best_idx+1]

        # ViTPose: get 133 whole-body keypoints for the best person
        det_array = np.concatenate([best_bbox, best_score[:, None]], axis=1)
        vitposes_out = vitpose_model.predict_pose(img_rgb, [det_array])

        if len(vitposes_out) == 0:
            raw_data.append(frame_data)
            continue

        keypoints_133 = vitposes_out[0]['keypoints']  # (133, 3)
        left_hand_kp = keypoints_133[-42:-21]   # (21, 3)
        right_hand_kp = keypoints_133[-21:]      # (21, 3)
        img_shape = img_cv2.shape

        for kp, hand_name in [(left_hand_kp, 'left'), (right_hand_kp, 'right')]:
            valid = kp[:, 2] > keypoint_conf_threshold
            if valid.sum() > min_valid_keypoints:
                valid_kp = kp[valid]
                raw_bbox = np.array([
                    valid_kp[:, 0].min(), valid_kp[:, 1].min(),
                    valid_kp[:, 0].max(), valid_kp[:, 1].max()
                ])
                bbox = enlarge_bbox(raw_bbox, scale=1.2, img_shape=img_shape)
                conf = float(valid_kp[:, 2].mean())

                frame_data[f'{hand_name}_bbox'] = bbox
                frame_data[f'{hand_name}_raw_bbox'] = raw_bbox
                frame_data[f'{hand_name}_keypoints'] = kp  # full (21, 3)
                frame_data[f'{hand_name}_conf'] = conf

        # Visualize
        if vis_dir is not None:
            img_vis = img_cv2.copy()
            if frame_data['left_bbox'] is not None:
                x1, y1, x2, y2 = frame_data['left_bbox'].astype(int)
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_vis, f"L vit (conf={frame_data['left_conf']:.2f})",
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if frame_data['right_bbox'] is not None:
                x1, y1, x2, y2 = frame_data['right_bbox'].astype(int)
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img_vis, f"R vit (conf={frame_data['right_conf']:.2f})",
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            # Draw hand keypoints + skeleton
            hand_edges = [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (0,9),(9,10),(10,11),(11,12),
                (0,13),(13,14),(14,15),(15,16),
                (0,17),(17,18),(18,19),(19,20),
            ]
            for hand_name, color in [('left', (0, 255, 0)), ('right', (255, 0, 0))]:
                kp = frame_data[f'{hand_name}_keypoints']
                if kp is None:
                    continue
                for j in range(kp.shape[0]):
                    x, y, c = kp[j]
                    if c > keypoint_conf_threshold:
                        cv2.circle(img_vis, (int(x), int(y)), 3, color, -1)
                for s, e in hand_edges:
                    if kp[s, 2] > keypoint_conf_threshold and kp[e, 2] > keypoint_conf_threshold:
                        pt1 = (int(kp[s, 0]), int(kp[s, 1]))
                        pt2 = (int(kp[e, 0]), int(kp[e, 1]))
                        cv2.line(img_vis, pt1, pt2, color, 1)
            img_fn = os.path.splitext(os.path.basename(img_path))[0]
            cv2.imwrite(os.path.join(pass1_vit_dir, f'{img_fn}_vitpose.jpg'), img_vis)

        raw_data.append(frame_data)

    # Statistics
    left_count = sum(1 for f in raw_data if f['left_bbox'] is not None)
    right_count = sum(1 for f in raw_data if f['right_bbox'] is not None)
    print(f"\nViTPose extracted bboxes for {len(raw_data)} frames:")
    print(f"  Left hand detected in {left_count} frames ({100*left_count/len(raw_data):.1f}%)")
    print(f"  Right hand detected in {right_count} frames ({100*right_count/len(raw_data):.1f}%)")

    if vis_dir is not None:
        video_path = os.path.join(vis_dir, 'pass1_raw_bboxes_vitpose.mp4')
        create_video_from_images(pass1_vit_dir, video_path, fps=fps)

    return raw_data


def merge_detections(yolo_data, vitpose_data):
    """
    Merge YOLO and ViTPose detections: per hand, pick the higher-confidence detection.
    Ties go to YOLO (existing behavior).
    """
    print("\n" + "="*80)
    print("PASS 1 Merge: Combining YOLO and ViTPose detections (per-hand best confidence)")
    print("="*80)

    merged_data = []
    yolo_wins = {'left': 0, 'right': 0}
    vit_wins = {'left': 0, 'right': 0}

    for yolo_frame, vit_frame in zip(yolo_data, vitpose_data):
        merged_frame = {
            'frame_idx': yolo_frame['frame_idx'],
            'img_path': yolo_frame['img_path'],
        }

        for hand in ['left', 'right']:
            bbox_key = f'{hand}_bbox'
            raw_bbox_key = f'{hand}_raw_bbox'
            keyp_key = f'{hand}_keypoints'
            conf_key = f'{hand}_conf'

            yolo_conf = yolo_frame[conf_key]
            vit_conf = vit_frame[conf_key]

            if vit_conf > yolo_conf:
                source = vit_frame
                if vit_conf > 0:
                    vit_wins[hand] += 1
            else:
                source = yolo_frame
                if yolo_conf > 0:
                    yolo_wins[hand] += 1

            merged_frame[bbox_key] = source[bbox_key]
            merged_frame[raw_bbox_key] = source[raw_bbox_key]
            merged_frame[keyp_key] = source[keyp_key]
            merged_frame[conf_key] = source[conf_key]

        merged_data.append(merged_frame)

    total = len(merged_data)
    print(f"  Left hand:  YOLO won {yolo_wins['left']}, ViTPose won {vit_wins['left']} "
          f"(of {total} frames)")
    print(f"  Right hand: YOLO won {yolo_wins['right']}, ViTPose won {vit_wins['right']} "
          f"(of {total} frames)")

    return merged_data


# ============================================================================
# PASS 2: Clean bbox sequences globally
# ============================================================================

def detect_overlapping_bboxes(raw_data, iou_threshold=0.7, containment_threshold=0.7):
    """
    Detect and remove overlapping bboxes (hallucinations) - YOLO version without keypoints.
    Uses simplified criteria:
    1. High bbox IoU (> 0.7) - very overlapping bboxes
    2. High containment (> 0.7) - one bbox is largely contained in another
    
    Since YOLO doesn't provide keypoints, we only use bbox-based overlap detection.
    This is actually more reliable since YOLO is specifically trained for hand detection.
    
    Also tracks which hand was removed due to overlap and which hand "caused" the removal,
    so we can potentially restore the removed hand if the winner is later invalidated.
    """
    print("\n" + "-"*80)
    print("Step 1: Detecting overlapping bboxes (hallucinations)")
    print("-"*80)
    
    overlap_count = 0
    
    for frame_data in raw_data:
        # Initialize removal tracking flags
        frame_data['left_removed_due_to_overlap_with'] = None
        frame_data['right_removed_due_to_overlap_with'] = None
        if frame_data['left_bbox'] is not None and frame_data['right_bbox'] is not None:
            # Criterion 1: Bbox IoU
            iou = compute_iou(frame_data['left_bbox'], frame_data['right_bbox'])
            
            # Criterion 2: Containment ratio (either bbox contains the other)
            left_in_right = compute_containment_ratio(frame_data['left_bbox'], frame_data['right_bbox'])
            right_in_left = compute_containment_ratio(frame_data['right_bbox'], frame_data['left_bbox'])
            max_containment = max(left_in_right, right_in_left)
            
            # Determine if this is an overlap/hallucination
            is_overlap = False
            reason = []
            
            if iou > iou_threshold:
                is_overlap = True
                reason.append(f"IoU={iou:.3f}")
            
            if max_containment > containment_threshold:
                is_overlap = True
                which_contained = "L_in_R" if left_in_right > right_in_left else "R_in_L"
                reason.append(f"containment={max_containment:.2f} ({which_contained})")
            
            if is_overlap:
                overlap_count += 1
                print(f"  Frame {frame_data['frame_idx']:04d}: {', '.join(reason)} - OVERLAP DETECTED")
                
                # Keep hand with higher confidence, track which hand was removed and why
                if frame_data['left_conf'] > frame_data['right_conf']:
                    print(f"    → Keeping LEFT (conf={frame_data['left_conf']:.3f}), removing RIGHT")
                    frame_data['right_bbox'] = None
                    frame_data['right_raw_bbox'] = None
                    frame_data['right_keypoints'] = None
                    frame_data['right_conf'] = 0.0
                    frame_data['right_removed_due_to_overlap_with'] = 'left'
                else:
                    print(f"    → Keeping RIGHT (conf={frame_data['right_conf']:.3f}), removing LEFT")
                    frame_data['left_bbox'] = None
                    frame_data['left_raw_bbox'] = None
                    frame_data['left_keypoints'] = None
                    frame_data['left_conf'] = 0.0
                    frame_data['left_removed_due_to_overlap_with'] = 'right'
    
    print(f"Removed overlapping bboxes in {overlap_count} frames")
    return raw_data


def fix_handedness_swaps_by_trajectory(raw_data, position_threshold=200, relaxed=True):
    """
    Detect handedness swaps by analyzing trajectory jumps.
    If one hand suddenly jumps to where the other hand was, it's likely mislabeled.

    Example: Green curve (right) drops to blue curve's position while blue disappears
    → The "right" detection is actually the left hand with wrong label

    Args:
        raw_data: List of per-frame detection dicts.
        position_threshold: Max center-x distance (px) to consider positions "close".
        relaxed: If True, also handle cases where prev frame has only one hand
                 (Case 3/4). Set to False to revert to original strict behavior
                 that requires both hands present in the previous frame.
    """
    print("\n" + "-"*80)
    print(f"Step 1.4a: Detecting handedness swaps via trajectory analysis (relaxed={relaxed})")
    print("-"*80)

    # Majority guard for relaxed cases: prevent cascade swaps that convert
    # the dominant hand label to the minority label.
    # Computed once from the ORIGINAL labels before any swaps in this step.
    if relaxed:
        left_count = sum(1 for d in raw_data if d['left_bbox'] is not None)
        right_count = sum(1 for d in raw_data if d['right_bbox'] is not None)
        total = left_count + right_count
        majority_ratio = 0.7
        majority_hand = None
        if total > 0:
            if left_count / total > majority_ratio:
                majority_hand = 'left'
            elif right_count / total > majority_ratio:
                majority_hand = 'right'
        print(f"  Majority guard: L={left_count}, R={right_count}, majority={majority_hand}")

    swap_count = 0
    n = len(raw_data)

    for i in range(1, n):
        left_bbox = raw_data[i]['left_bbox']
        right_bbox = raw_data[i]['right_bbox']
        prev_left = raw_data[i-1]['left_bbox']
        prev_right = raw_data[i-1]['right_bbox']

        # Case 1: Right hand jumps to where left hand was (left disappears)
        # Requires both hands in prev frame to compare distances
        if right_bbox is not None and left_bbox is None and prev_left is not None and prev_right is not None:
            # Calculate positions
            prev_left_cx = (prev_left[0] + prev_left[2]) / 2
            prev_right_cx = (prev_right[0] + prev_right[2]) / 2
            curr_right_cx = (right_bbox[0] + right_bbox[2]) / 2

            # Distance from current "right" to previous left/right
            dist_to_prev_left = abs(curr_right_cx - prev_left_cx)
            dist_to_prev_right = abs(curr_right_cx - prev_right_cx)

            # If current "right" is much closer to previous left than previous right, it's swapped
            if dist_to_prev_left < dist_to_prev_right and dist_to_prev_right > position_threshold:
                print(f"  Frame {i:04d}: RIGHT jumped to LEFT position (dist to prev_left={dist_to_prev_left:.0f}, dist to prev_right={dist_to_prev_right:.0f}) - Swapping RIGHT → LEFT")
                raw_data[i]['left_bbox'] = right_bbox
                raw_data[i]['left_raw_bbox'] = raw_data[i]['right_raw_bbox']
                raw_data[i]['left_keypoints'] = raw_data[i]['right_keypoints']
                raw_data[i]['left_conf'] = raw_data[i]['right_conf']
                raw_data[i]['right_bbox'] = None
                raw_data[i]['right_raw_bbox'] = None
                raw_data[i]['right_keypoints'] = None
                raw_data[i]['right_conf'] = 0.0
                swap_count += 1

        # Case 2: Left hand jumps to where right hand was (right disappears)
        # Requires both hands in prev frame to compare distances
        elif left_bbox is not None and right_bbox is None and prev_right is not None and prev_left is not None:
            # Calculate positions
            prev_left_cx = (prev_left[0] + prev_left[2]) / 2
            prev_right_cx = (prev_right[0] + prev_right[2]) / 2
            curr_left_cx = (left_bbox[0] + left_bbox[2]) / 2

            # Distance from current "left" to previous left/right
            dist_to_prev_left = abs(curr_left_cx - prev_left_cx)
            dist_to_prev_right = abs(curr_left_cx - prev_right_cx)

            # If current "left" is much closer to previous right than previous left, it's swapped
            if dist_to_prev_right < dist_to_prev_left and dist_to_prev_left > position_threshold:
                print(f"  Frame {i:04d}: LEFT jumped to RIGHT position (dist to prev_right={dist_to_prev_right:.0f}, dist to prev_left={dist_to_prev_left:.0f}) - Swapping LEFT → RIGHT")
                raw_data[i]['right_bbox'] = left_bbox
                raw_data[i]['right_raw_bbox'] = raw_data[i]['left_raw_bbox']
                raw_data[i]['right_keypoints'] = raw_data[i]['left_keypoints']
                raw_data[i]['right_conf'] = raw_data[i]['left_conf']
                raw_data[i]['left_bbox'] = None
                raw_data[i]['left_raw_bbox'] = None
                raw_data[i]['left_keypoints'] = None
                raw_data[i]['left_conf'] = 0.0
                swap_count += 1

        # --- Relaxed cases: prev frame only has ONE hand ---
        # These handle situations where overlap detection already removed one
        # hand in the previous frame, leaving no reference for Cases 1/2.
        # Majority guard: don't swap the majority hand to minority (prevents cascade).
        elif relaxed:
            # Case 3: Prev only has LEFT, curr only has RIGHT, positions close → swap R→L
            if (right_bbox is not None and left_bbox is None
                    and prev_left is not None and prev_right is None):
                # Majority guard: this would swap right→left; block if right is majority
                if majority_hand == 'right':
                    pass
                else:
                    prev_left_cx = (prev_left[0] + prev_left[2]) / 2
                    curr_right_cx = (right_bbox[0] + right_bbox[2]) / 2
                    dist = abs(curr_right_cx - prev_left_cx)

                    if dist < position_threshold:
                        print(f"  Frame {i:04d}: Prev only LEFT (cx={prev_left_cx:.0f}), curr only RIGHT (cx={curr_right_cx:.0f}), "
                              f"dist={dist:.0f} < {position_threshold} - Swapping RIGHT → LEFT")
                        raw_data[i]['left_bbox'] = right_bbox
                        raw_data[i]['left_raw_bbox'] = raw_data[i].get('right_raw_bbox')
                        raw_data[i]['left_keypoints'] = raw_data[i]['right_keypoints']
                        raw_data[i]['left_conf'] = raw_data[i]['right_conf']
                        raw_data[i]['right_bbox'] = None
                        raw_data[i]['right_raw_bbox'] = None
                        raw_data[i]['right_keypoints'] = None
                        raw_data[i]['right_conf'] = 0.0
                        swap_count += 1

            # Case 4: Prev only has RIGHT, curr only has LEFT, positions close → swap L→R
            elif (left_bbox is not None and right_bbox is None
                      and prev_right is not None and prev_left is None):
                # Majority guard: this would swap left→right; block if left is majority
                if majority_hand == 'left':
                    pass
                else:
                    prev_right_cx = (prev_right[0] + prev_right[2]) / 2
                    curr_left_cx = (left_bbox[0] + left_bbox[2]) / 2
                    dist = abs(curr_left_cx - prev_right_cx)

                    if dist < position_threshold:
                        print(f"  Frame {i:04d}: Prev only RIGHT (cx={prev_right_cx:.0f}), curr only LEFT (cx={curr_left_cx:.0f}), "
                              f"dist={dist:.0f} < {position_threshold} - Swapping LEFT → RIGHT")
                        raw_data[i]['right_bbox'] = left_bbox
                        raw_data[i]['right_raw_bbox'] = raw_data[i].get('left_raw_bbox')
                        raw_data[i]['right_keypoints'] = raw_data[i]['left_keypoints']
                        raw_data[i]['right_conf'] = raw_data[i]['left_conf']
                        raw_data[i]['left_bbox'] = None
                        raw_data[i]['left_raw_bbox'] = None
                        raw_data[i]['left_keypoints'] = None
                        raw_data[i]['left_conf'] = 0.0
                    swap_count += 1

    if swap_count > 0:
        print(f"  Fixed {swap_count} trajectory-based handedness swaps")
    else:
        print(f"  No trajectory-based swaps detected")

    return raw_data


def fix_handedness_swaps(raw_data, context_window=10):
    """
    Detect and fix handedness swaps where YOLO misclassifies hand labels.
    For example, detecting left hand as right hand for a few frames.
    
    Strategy: If a hand appears/disappears while the other hand simultaneously 
    disappears/appears in a short sequence, it's likely a handedness swap.
    """
    print("\n" + "-"*80)
    print(f"Step 1.4: Detecting handedness swaps (context window={context_window})")
    print("-"*80)
    
    swap_count = 0
    n = len(raw_data)
    
    for i in range(n):
        left_bbox = raw_data[i]['left_bbox']
        right_bbox = raw_data[i]['right_bbox']
        
        # Case 1: Only right hand detected, but neighbors mostly have only left hand
        if right_bbox is not None and left_bbox is None:
            # Count neighbors with left/right
            left_count = 0
            right_count = 0
            
            for j in range(max(0, i - context_window), min(n, i + context_window + 1)):
                if j == i:
                    continue
                if raw_data[j]['left_bbox'] is not None and raw_data[j]['right_bbox'] is None:
                    left_count += 1
                elif raw_data[j]['right_bbox'] is not None and raw_data[j]['left_bbox'] is None:
                    right_count += 1
            
            # If neighbors are mostly left-only, this right is probably mislabeled left
            if left_count > 0 and left_count > right_count * 2:
                print(f"  Frame {i:04d}: Swapping RIGHT → LEFT (neighbors: {left_count} left-only, {right_count} right-only)")
                raw_data[i]['left_bbox'] = right_bbox
                raw_data[i]['left_raw_bbox'] = raw_data[i]['right_raw_bbox']
                raw_data[i]['left_keypoints'] = raw_data[i]['right_keypoints']
                raw_data[i]['left_conf'] = raw_data[i]['right_conf']
                raw_data[i]['right_bbox'] = None
                raw_data[i]['right_raw_bbox'] = None
                raw_data[i]['right_keypoints'] = None
                raw_data[i]['right_conf'] = 0.0
                swap_count += 1

        # Case 2: Only left hand detected, but neighbors mostly have only right hand
        elif left_bbox is not None and right_bbox is None:
            # Count neighbors with left/right
            left_count = 0
            right_count = 0

            for j in range(max(0, i - context_window), min(n, i + context_window + 1)):
                if j == i:
                    continue
                if raw_data[j]['left_bbox'] is not None and raw_data[j]['right_bbox'] is None:
                    left_count += 1
                elif raw_data[j]['right_bbox'] is not None and raw_data[j]['left_bbox'] is None:
                    right_count += 1

            # If neighbors are mostly right-only, this left is probably mislabeled right
            if right_count > 0 and right_count > left_count * 2:
                print(f"  Frame {i:04d}: Swapping LEFT → RIGHT (neighbors: {left_count} left-only, {right_count} right-only)")
                raw_data[i]['right_bbox'] = left_bbox
                raw_data[i]['right_raw_bbox'] = raw_data[i]['left_raw_bbox']
                raw_data[i]['right_keypoints'] = raw_data[i]['left_keypoints']
                raw_data[i]['right_conf'] = raw_data[i]['left_conf']
                raw_data[i]['left_bbox'] = None
                raw_data[i]['left_raw_bbox'] = None
                raw_data[i]['left_keypoints'] = None
                raw_data[i]['left_conf'] = 0.0
                swap_count += 1
    
    if swap_count > 0:
        print(f"  Fixed {swap_count} handedness swaps")
    else:
        print(f"  No handedness swaps detected")
    
    return raw_data


def fix_handedness_swaps_frame_to_frame(raw_data, iou_threshold=0.7, max_gap=10):
    """
    Detect frame-to-frame handedness swaps: same bbox position but different label.

    Example problem:
    - Frame N:   Right hand at position X (green box)
    - Frame N+1: Left hand at position X (blue box) - SAME BBOX, WRONG LABEL!

    This catches YOLO randomly mislabeling the same physical hand.

    Strategy:
    If a hand suddenly appears while the other hand recently disappeared (within max_gap frames),
    AND the new hand's bbox has high IoU with the disappeared hand's LAST VALID bbox,
    → It's the same hand with wrong label! Swap it back!

    IMPORTANT: Uses a snapshot of the original labels for lookback to prevent cascade errors.
    Without this, swapping frame N causes frame N+1 to see the swapped result, which triggers
    another swap, cascading through the entire sequence.

    Args:
        max_gap: Maximum frames to look back for last valid other hand (default 10)
    """
    print("\n" + "-"*80)
    print(f"Step 1.4: Detecting frame-to-frame handedness swaps (IoU threshold={iou_threshold}, max_gap={max_gap})")
    print("-"*80)

    # Snapshot labels BEFORE any swaps — lookback reads from this to prevent cascade
    snapshot = [{
        'left_bbox': f['left_bbox'].copy() if f['left_bbox'] is not None else None,
        'right_bbox': f['right_bbox'].copy() if f['right_bbox'] is not None else None,
    } for f in raw_data]

    # Global majority vote: count single-hand frames to determine dominant label
    # This prevents minority mislabels from cascading into the majority
    left_only_count = sum(1 for f in raw_data if f['left_bbox'] is not None and f['right_bbox'] is None)
    right_only_count = sum(1 for f in raw_data if f['right_bbox'] is not None and f['left_bbox'] is None)
    both_count = sum(1 for f in raw_data if f['left_bbox'] is not None and f['right_bbox'] is not None)
    total_single = left_only_count + right_only_count
    print(f"  Label distribution: left_only={left_only_count}, right_only={right_only_count}, both={both_count}")

    # Determine if there's a clear majority label (>70% of single-hand frames)
    # If so, block swaps that convert majority→minority
    majority_hand = None
    if total_single > 0:
        left_ratio = left_only_count / total_single
        right_ratio = right_only_count / total_single
        if right_ratio > 0.7:
            majority_hand = 'right'
            print(f"  Majority hand: RIGHT ({right_ratio:.1%} of single-hand frames)")
        elif left_ratio > 0.7:
            majority_hand = 'left'
            print(f"  Majority hand: LEFT ({left_ratio:.1%} of single-hand frames)")

    swap_count = 0

    for i in range(1, len(raw_data)):
        curr = raw_data[i]

        for hand_name in ['left', 'right']:
            bbox_key = f'{hand_name}_bbox'
            keyp_key = f'{hand_name}_keypoints'
            conf_key = f'{hand_name}_conf'
            other_hand = 'right' if hand_name == 'left' else 'left'
            other_bbox_key = f'{other_hand}_bbox'
            other_keyp_key = f'{other_hand}_keypoints'
            other_conf_key = f'{other_hand}_conf'

            curr_bbox = curr[bbox_key]
            curr_other_bbox = curr[other_bbox_key]

            # Case: Current hand exists and other hand is missing
            if curr_bbox is not None and curr_other_bbox is None:
                # Look back in SNAPSHOT (not live data) to find last valid OTHER hand
                last_valid_other_bbox = None
                last_valid_other_idx = None

                for j in range(i - 1, max(0, i - max_gap - 1), -1):
                    if snapshot[j][other_bbox_key] is not None:
                        last_valid_other_bbox = snapshot[j][other_bbox_key]
                        last_valid_other_idx = j
                        break

                if last_valid_other_bbox is not None:
                    gap = i - last_valid_other_idx
                    iou = compute_iou(curr_bbox, last_valid_other_bbox)
                    if iou > iou_threshold:
                        # Guard: block swaps that convert majority label → minority label
                        # e.g., if 90% frames are "right", don't swap right→left based on
                        # a few mislabeled "left" frames in the lookback window
                        if majority_hand is not None and hand_name == majority_hand:
                            print(f"  Frame {curr['frame_idx']:04d}: {hand_name} matches {other_hand}'s position "
                                  f"(IoU={iou:.3f}) but BLOCKED — would swap majority '{majority_hand}' to minority '{other_hand}'")
                            continue

                        print(f"  Frame {curr['frame_idx']:04d}: {hand_name} at same position as {other_hand} "
                              f"(last seen at frame {raw_data[last_valid_other_idx]['frame_idx']}, gap={gap}, IoU={iou:.3f}) "
                              f"- SWAPPING {hand_name} → {other_hand}")

                        # Swap: this is actually the OTHER hand
                        other_raw_bbox_key = f'{other_hand}_raw_bbox'
                        raw_bbox_key = f'{hand_name}_raw_bbox'
                        curr[other_bbox_key] = curr_bbox.copy()
                        curr[other_raw_bbox_key] = curr[raw_bbox_key].copy() if curr.get(raw_bbox_key) is not None else None
                        curr[other_keyp_key] = curr[keyp_key].copy() if curr[keyp_key] is not None else None
                        curr[other_conf_key] = curr[conf_key]

                        # Remove from current hand
                        curr[bbox_key] = None
                        curr[raw_bbox_key] = None
                        curr[keyp_key] = None
                        curr[conf_key] = 0.0

                        swap_count += 1

    if swap_count > 0:
        print(f"  Fixed {swap_count} frame-to-frame handedness swaps")
    else:
        print(f"  No frame-to-frame swaps detected")

    return raw_data


def fix_handedness_inconsistencies(raw_data, original_data, context_window=5):
    """
    Detect handedness swaps using SPATIAL-TEMPORAL consistency.
    
    Example problem: [L, L, L, _, _, R, R, R, L, L]
    - The 3 Rs are spatially in the LEFT hand's trajectory but mislabeled as RIGHT
    
    Strategy:
    1. For each detection, compute IoU with recent detections of SAME hand
    2. Also compute IoU with recent detections of OTHER hand
    3. If IoU with OTHER hand's trajectory is much higher → handedness is swapped!
    """
    print("\n" + "-"*80)
    print(f"Step 1.5: Fixing handedness inconsistencies via spatial-temporal consistency (window={context_window})")
    print("-"*80)
    
    swap_count = 0
    
    for i, frame_data in enumerate(raw_data):
        for hand_name in ['left', 'right']:
            bbox_key = f'{hand_name}_bbox'
            keyp_key = f'{hand_name}_keypoints'
            conf_key = f'{hand_name}_conf'
            other_hand = 'right' if hand_name == 'left' else 'left'
            other_bbox_key = f'{other_hand}_bbox'
            other_keyp_key = f'{other_hand}_keypoints'
            other_conf_key = f'{other_hand}_conf'
            
            current_bbox = frame_data[bbox_key]
            if current_bbox is None:
                continue
            
            # Collect recent bboxes for SAME hand and OTHER hand
            same_hand_bboxes = []
            other_hand_bboxes = []
            
            # Look at context window (both past and future)
            for j in range(max(0, i - context_window), min(len(raw_data), i + context_window + 1)):
                if j == i:
                    continue  # Skip current frame
                
                # Collect SAME hand bboxes
                if raw_data[j][bbox_key] is not None:
                    same_hand_bboxes.append(raw_data[j][bbox_key])
                
                # Collect OTHER hand bboxes
                if raw_data[j][other_bbox_key] is not None:
                    other_hand_bboxes.append(raw_data[j][other_bbox_key])
            
            # Need enough context to make a decision
            if len(same_hand_bboxes) < 2 and len(other_hand_bboxes) < 2:
                continue  # Not enough spatial context
            
            # Compute average IoU with SAME hand trajectory
            same_hand_ious = []
            for bbox in same_hand_bboxes:
                iou = compute_iou(current_bbox, bbox)
                same_hand_ious.append(iou)
            avg_iou_same = np.mean(same_hand_ious) if len(same_hand_ious) > 0 else 0.0
            
            # Compute average IoU with OTHER hand trajectory
            other_hand_ious = []
            for bbox in other_hand_bboxes:
                iou = compute_iou(current_bbox, bbox)
                other_hand_ious.append(iou)
            avg_iou_other = np.mean(other_hand_ious) if len(other_hand_ious) > 0 else 0.0
            
            # DETECTION LOGIC: If current bbox fits OTHER hand's trajectory much better, it's swapped!
            # Criteria:
            # 1. OTHER hand trajectory has high overlap (>0.5) with current bbox
            # 2. OTHER hand trajectory overlap is significantly higher than SAME hand (1.5x)
            # 3. We have enough OTHER hand detections nearby (at least 3)
            if (len(other_hand_bboxes) >= 3 and 
                avg_iou_other > 0.5 and 
                avg_iou_other > avg_iou_same * 1.5):
                
                print(f"  Frame {frame_data['frame_idx']:04d} ({hand_name}): "
                      f"Bbox fits {other_hand} trajectory better! "
                      f"(IoU with {hand_name}={avg_iou_same:.3f}, IoU with {other_hand}={avg_iou_other:.3f}, "
                      f"neighbors: {len(same_hand_bboxes)} {hand_name}, {len(other_hand_bboxes)} {other_hand}) "
                      f"- SWAPPING {hand_name} → {other_hand}")
                
                # Swap: move current bbox to OTHER hand
                other_raw_bbox_key = f'{other_hand}_raw_bbox'
                raw_bbox_key = f'{hand_name}_raw_bbox'
                frame_data[other_bbox_key] = current_bbox.copy()
                frame_data[other_raw_bbox_key] = frame_data[raw_bbox_key].copy() if frame_data.get(raw_bbox_key) is not None else None
                frame_data[other_keyp_key] = frame_data[keyp_key].copy() if frame_data[keyp_key] is not None else None
                frame_data[other_conf_key] = frame_data[conf_key]

                # Remove from current hand
                frame_data[bbox_key] = None
                frame_data[raw_bbox_key] = None
                frame_data[keyp_key] = None
                frame_data[conf_key] = 0.0
                
                swap_count += 1
    
    if swap_count > 0:
        print(f"  Fixed {swap_count} handedness swaps via spatial-temporal consistency")
    else:
        print(f"  No spatial-temporal inconsistencies found")
    
    return raw_data


def detect_bbox_jumps(raw_data, base_iou_threshold=0.5):
    """
    Detect and fix bbox jumps with ADAPTIVE threshold based on detection gaps.
    
    Adaptive threshold formula: adjusted_threshold = max(0.0, base_threshold - gap_frames * 0.15)
    - Gap 0 (consecutive): threshold = 0.5 (strict, catches hallucinations)
    - Gap 1: threshold = 0.35 (slightly lenient)
    - Gap 2: threshold = 0.2 (more lenient)
    - Gap 3+: threshold = 0.05/0.0 (very lenient, accepts genuine movement)
    
    This allows genuine hand movement after detection gaps while still protecting
    against spurious jumps in consecutive frames.
    """
    print("\n" + "-"*80)
    print(f"Step 2: Detecting bbox jumps with adaptive threshold (base={base_iou_threshold})")
    print("-"*80)
    
    for hand_name in ['left', 'right']:
        bbox_key = f'{hand_name}_bbox'
        hand_id = 0 if hand_name == 'left' else 1
        
        last_valid_bbox = None
        last_valid_idx = None
        frames_since_last_detection = 0
        jump_count = 0
        
        for i, frame_data in enumerate(raw_data):
            current_bbox = frame_data[bbox_key]
            
            if current_bbox is not None:
                if last_valid_bbox is not None:
                    # Calculate IoU with last valid bbox
                    iou = compute_iou(current_bbox, last_valid_bbox)
                    
                    # Adaptive threshold based on gap
                    gap_frames = frames_since_last_detection
                    adjusted_threshold = max(0.0, base_iou_threshold - (gap_frames * 0.01))
                    
                    if iou < adjusted_threshold:
                        # Jump detected!
                        jump_count += 1
                        print(f"  Frame {frame_data['frame_idx']:04d} ({hand_name}): "
                              f"IoU={iou:.3f} < threshold={adjusted_threshold:.3f} (gap={gap_frames}) - JUMP DETECTED")
                        print(f"    → Current bbox: [{current_bbox[0]:.1f}, {current_bbox[1]:.1f}, {current_bbox[2]:.1f}, {current_bbox[3]:.1f}]")
                        print(f"    → Last valid bbox (frame {last_valid_idx}): [{last_valid_bbox[0]:.1f}, {last_valid_bbox[1]:.1f}, {last_valid_bbox[2]:.1f}, {last_valid_bbox[3]:.1f}]")
                        print(f"    → Replacing with last valid bbox from frame {last_valid_idx}")
                        frame_data[bbox_key] = last_valid_bbox.copy()
                    else:
                        # Valid detection
                        if gap_frames > 0:
                            print(f"  Frame {frame_data['frame_idx']:04d} ({hand_name}): "
                                  f"IoU={iou:.3f} > threshold={adjusted_threshold:.3f} (gap={gap_frames}) - ACCEPTING movement")
                        last_valid_bbox = current_bbox.copy()
                        last_valid_idx = i
                        frames_since_last_detection = 0
                else:
                    # First valid bbox
                    last_valid_bbox = current_bbox.copy()
                    last_valid_idx = i
                    frames_since_last_detection = 0
            else:
                # Hand is missing
                frames_since_last_detection += 1
        
        if jump_count > 0:
            print(f"  {hand_name.capitalize()}: Fixed {jump_count} jumps")
    
    return raw_data


def interpolate_missing_bboxes(raw_data, max_gap=5):
    """Interpolate missing bboxes for short gaps and create dummy keypoints."""
    print("\n" + "-"*80)
    print(f"Step 3: Interpolating missing bboxes (max gap={max_gap} frames)")
    print("-"*80)
    
    for hand_name in ['left', 'right']:
        bbox_key = f'{hand_name}_bbox'
        keyp_key = f'{hand_name}_keypoints'
        conf_key = f'{hand_name}_conf'
        
        # Find valid bbox indices
        valid_indices = [i for i, f in enumerate(raw_data) if f[bbox_key] is not None]
        
        if len(valid_indices) < 2:
            continue
        
        interpolated_count = 0
        
        for i in range(len(valid_indices) - 1):
            start_idx = valid_indices[i]
            end_idx = valid_indices[i + 1]
            gap = end_idx - start_idx - 1
            
            if 0 < gap <= max_gap:
                start_bbox = raw_data[start_idx][bbox_key]
                end_bbox = raw_data[end_idx][bbox_key]
                
                for j in range(1, gap + 1):
                    alpha = j / (gap + 1)
                    interp_bbox = (1 - alpha) * start_bbox + alpha * end_bbox
                    raw_data[start_idx + j][bbox_key] = interp_bbox
                    
                    # Create dummy keypoints for interpolated frames
                    dummy_keypoints = np.zeros((21, 3))
                    dummy_keypoints[:, 2] = 0.5
                    raw_data[start_idx + j][keyp_key] = dummy_keypoints
                    raw_data[start_idx + j][conf_key] = 0.5
                    
                    interpolated_count += 1
                
                print(f"  {hand_name.capitalize()}: Interpolated frames {start_idx+1}-{end_idx-1} (gap={gap})")
        
        if interpolated_count > 0:
            print(f"  {hand_name.capitalize()}: Interpolated {interpolated_count} frames total")
    
    return raw_data


def smooth_bbox_sequence(raw_data, window_size=5):
    """Apply temporal smoothing using moving average."""
    print("\n" + "-"*80)
    print(f"Step 4: Smoothing bbox sequences (window={window_size})")
    print("-"*80)
    
    for hand_name in ['left', 'right']:
        bbox_key = f'{hand_name}_bbox'
        
        bboxes = [f[bbox_key] for f in raw_data]
        smoothed_bboxes = []
        
        for i in range(len(bboxes)):
            if bboxes[i] is None:
                smoothed_bboxes.append(None)
                continue
            
            window_start = max(0, i - window_size // 2)
            window_end = min(len(bboxes), i + window_size // 2 + 1)
            window_bboxes = [bboxes[j] for j in range(window_start, window_end) if bboxes[j] is not None]
            
            if len(window_bboxes) > 0:
                smoothed_bbox = np.mean(window_bboxes, axis=0)
                smoothed_bboxes.append(smoothed_bbox)
            else:
                smoothed_bboxes.append(bboxes[i])
        
        for i, frame_data in enumerate(raw_data):
            if smoothed_bboxes[i] is not None:
                frame_data[bbox_key] = smoothed_bboxes[i]
    
    print("  Smoothing complete")
    return raw_data


def check_if_bboxes_overlap(bbox1, bbox2, keypoints1=None, keypoints2=None, 
                            iou_threshold=0.7, containment_threshold=0.7, 
                            wrist_dist_ratio=0.15, keypoint_sim_ratio=0.10):
    """
    Check if two bboxes overlap using the same comprehensive criteria as detect_overlapping_bboxes.
    Returns True if the bboxes are considered overlapping (hallucination).
    """
    # Compute average bbox size for normalization
    size1 = max(bbox1[2] - bbox1[0], bbox1[3] - bbox1[1])
    size2 = max(bbox2[2] - bbox2[0], bbox2[3] - bbox2[1])
    avg_bbox_size = (size1 + size2) / 2
    
    # Criterion 1: Bbox IoU
    iou = compute_iou(bbox1, bbox2)
    if iou > iou_threshold:
        return True
    
    # Criterion 2: Containment ratio
    containment_1_in_2 = compute_containment_ratio(bbox1, bbox2)
    containment_2_in_1 = compute_containment_ratio(bbox2, bbox1)
    max_containment = max(containment_1_in_2, containment_2_in_1)
    if max_containment > containment_threshold:
        return True
    
    # Criterion 3 & 4: Keypoint-based checks (if keypoints available)
    if keypoints1 is not None and keypoints2 is not None:
        # Wrist distance
        wrist1 = keypoints1[0, :2]
        wrist2 = keypoints2[0, :2]
        wrist_dist = np.linalg.norm(wrist1 - wrist2)
        wrist_dist_normalized = wrist_dist / avg_bbox_size
        if wrist_dist_normalized < wrist_dist_ratio:
            return True
        
        # Average keypoint distance
        keypoint_dists = []
        for i in range(min(len(keypoints1), len(keypoints2))):
            kp1 = keypoints1[i, :2]
            kp2 = keypoints2[i, :2]
            dist = np.linalg.norm(kp1 - kp2)
            keypoint_dists.append(dist)
        avg_keypoint_dist = np.mean(keypoint_dists)
        avg_keypoint_dist_normalized = avg_keypoint_dist / avg_bbox_size
        if avg_keypoint_dist_normalized < keypoint_sim_ratio:
            return True
    
    return False


def plot_bbox_trajectories(raw_data, vis_dir, filename='bbox_trajectories.png'):
    """
    Plot left and right hand bbox center trajectories over time.
    This helps visualize handedness consistency and detect label swaps.
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    
    frames = []
    left_centers_x = []
    left_centers_y = []
    right_centers_x = []
    right_centers_y = []
    
    for i, frame_data in enumerate(raw_data):
        frames.append(i)
        
        # Left hand
        if frame_data['left_bbox'] is not None:
            left_cx = (frame_data['left_bbox'][0] + frame_data['left_bbox'][2]) / 2
            left_cy = (frame_data['left_bbox'][1] + frame_data['left_bbox'][3]) / 2
            left_centers_x.append(left_cx)
            left_centers_y.append(left_cy)
        else:
            left_centers_x.append(None)
            left_centers_y.append(None)
        
        # Right hand
        if frame_data['right_bbox'] is not None:
            right_cx = (frame_data['right_bbox'][0] + frame_data['right_bbox'][2]) / 2
            right_cy = (frame_data['right_bbox'][1] + frame_data['right_bbox'][3]) / 2
            right_centers_x.append(right_cx)
            right_centers_y.append(right_cy)
        else:
            right_centers_x.append(None)
            right_centers_y.append(None)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
    
    # Plot X coordinates (horizontal position)
    ax1.plot(frames, left_centers_x, 'g', label='Left hand', linewidth=1.5, alpha=0.8)
    ax1.plot(frames, right_centers_x, 'b', label='Right hand', linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('Frame', fontsize=12)
    ax1.set_ylabel('X coordinate (pixels)', fontsize=12)
    ax1.set_title('Hand Bbox Center X-Position Over Time\n(Left hand should be consistently < Right hand)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Highlight regions where left > right (potential swaps)
    swap_count = 0
    for i in range(len(frames)):
        if left_centers_x[i] is not None and right_centers_x[i] is not None:
            if left_centers_x[i] > right_centers_x[i]:
                ax1.axvspan(i-0.5, i+0.5, alpha=0.3, color='red')
                swap_count += 1
    
    # Plot Y coordinates (vertical position)
    ax2.plot(frames, left_centers_y, 'g', label='Left hand', linewidth=1.5, alpha=0.8)
    ax2.plot(frames, right_centers_y, 'b', label='Right hand', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('Frame', fontsize=12)
    ax2.set_ylabel('Y coordinate (pixels)', fontsize=12)
    ax2.set_title('Hand Bbox Center Y-Position Over Time', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()  # Invert Y axis since image coordinates go top-to-bottom
    
    plt.tight_layout()
    save_path = os.path.join(vis_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n{'='*80}")
    print(f"Bbox trajectory plot saved to: {save_path}")
    print(f"  Red regions indicate frames where left_x > right_x (potential label swaps)")
    print(f"  Total frames with left_x > right_x: {swap_count}")
    print(f"{'='*80}\n")


def clean_bbox_sequences(raw_data, vis_dir=None, fps=15):
    """Pass 2: Clean bbox sequences using global temporal information."""
    print("\n" + "="*80)
    print("PASS 2: Cleaning bbox sequences")
    print("="*80)
    
    if vis_dir is not None:
        pass2_dir = os.path.join(vis_dir, 'pass2_cleaned_bboxes')
        os.makedirs(pass2_dir, exist_ok=True)
    
    # Store original bboxes and keypoints for comparison and restoration
    original_data = [{
        'left_bbox': f['left_bbox'].copy() if f['left_bbox'] is not None else None,
        'right_bbox': f['right_bbox'].copy() if f['right_bbox'] is not None else None,
        'left_keypoints': f['left_keypoints'].copy() if f['left_keypoints'] is not None else None,
        'right_keypoints': f['right_keypoints'].copy() if f['right_keypoints'] is not None else None,
        'left_conf': f['left_conf'] if f['left_conf'] is not None else None,
        'right_conf': f['right_conf'] if f['right_conf'] is not None else None,
    } for f in raw_data]
    
    # Plot trajectories BEFORE cleaning
    if vis_dir is not None:
        plot_bbox_trajectories(raw_data, vis_dir, filename='bbox_trajectories_before_cleaning.png')
    
    # Remove oversized bboxes (> 50% of image area)
    print("\n" + "-"*80)
    print("Step 1.5: Removing oversized bboxes (> 50% of image area)")
    print("-"*80)
    MAX_BBOX_AREA_RATIO = 0.5
    removed_oversized = 0
    
    # Load first image to get dimensions (all images should have same size)
    first_img = cv2.imread(raw_data[0]['img_path'])
    img_area = first_img.shape[0] * first_img.shape[1]
    
    for frame_data in raw_data:
        for hand_name in ['left', 'right']:
            bbox_key = f'{hand_name}_bbox'
            keyp_key = f'{hand_name}_keypoints'
            conf_key = f'{hand_name}_conf'
            
            bbox = frame_data[bbox_key]
            if bbox is not None:
                bbox_width = bbox[2] - bbox[0]
                bbox_height = bbox[3] - bbox[1]
                bbox_area = bbox_width * bbox_height
                area_ratio = bbox_area / img_area
                
                if area_ratio > MAX_BBOX_AREA_RATIO:
                    print(f"  Frame {frame_data['frame_idx']:04d} ({hand_name}): "
                          f"Bbox too large ({area_ratio:.1%} of image) - REMOVING")
                    frame_data[bbox_key] = None
                    frame_data[f'{hand_name}_raw_bbox'] = None
                    frame_data[keyp_key] = None
                    frame_data[conf_key] = 0.0
                    removed_oversized += 1
    
    if removed_oversized > 0:
        print(f"  Removed {removed_oversized} oversized bboxes")
    else:
        print(f"  No oversized bboxes found")
    
    raw_data = detect_overlapping_bboxes(raw_data, iou_threshold=0.7)
    raw_data = fix_handedness_swaps_by_trajectory(raw_data, position_threshold=200)
    raw_data = fix_handedness_inconsistencies(raw_data, original_data, context_window=5)
    # raw_data = detect_bbox_jumps(raw_data, base_iou_threshold=0.4)
    raw_data = fix_handedness_swaps_frame_to_frame(raw_data, iou_threshold=0.6)
    
    # Apply patience mechanism with interpolation when hand reappears
    print("\n" + "-"*80)
    print("Step 3: Applying patience mechanism with interpolation")
    print("-"*80)
    PATIENCE_FRAMES = 25
    MAX_PATIENCE_WITHOUT_RETURN = 0  # No padding if hand never reappears
    patience_applied_count = 0
    interpolated_count = 0
    trimmed_count = 0
    
    for hand_id in [0, 1]:
        hand_name = 'left' if hand_id == 0 else 'right'
        bbox_key = f'{hand_name}_bbox'
        raw_bbox_key = f'{hand_name}_raw_bbox'
        keyp_key = f'{hand_name}_keypoints'
        conf_key = f'{hand_name}_conf'
        
        i = 0
        while i < len(raw_data):
            # Find start of a gap (hand missing)
            if raw_data[i][bbox_key] is None:
                # Look back to find last valid bbox
                last_valid_idx = None
                for j in range(i - 1, -1, -1):
                    if raw_data[j][bbox_key] is not None:
                        last_valid_idx = j
                        break
                
                if last_valid_idx is None:
                    i += 1
                    continue
                
                # Find end of gap (hand reappears or sequence ends)
                gap_start = i
                gap_end = None
                for j in range(i, min(i + PATIENCE_FRAMES, len(raw_data))):
                    if raw_data[j][bbox_key] is not None:
                        gap_end = j
                        break
                
                if gap_end is not None:
                    # Hand reappeared within patience threshold - CHECK DISTANCE FIRST
                    gap_length = gap_end - gap_start
                    start_bbox = raw_data[last_valid_idx][bbox_key]
                    end_bbox = raw_data[gap_end][bbox_key]
                    
                    # Compute center distance between start and end bbox
                    start_cx = (start_bbox[0] + start_bbox[2]) / 2
                    start_cy = (start_bbox[1] + start_bbox[3]) / 2
                    end_cx = (end_bbox[0] + end_bbox[2]) / 2
                    end_cy = (end_bbox[1] + end_bbox[3]) / 2
                    center_distance = np.sqrt((end_cx - start_cx)**2 + (end_cy - start_cy)**2)
                    
                    # Compute bbox width (use average of start and end)
                    start_width = start_bbox[2] - start_bbox[0]
                    end_width = end_bbox[2] - end_bbox[0]
                    avg_width = (start_width + end_width) / 2
                    
                    # Only interpolate if bboxes are close enough (within 2× bbox width)
                    max_distance = avg_width
                    
                    if center_distance <= max_distance:
                        # Bboxes are close - INTERPOLATE
                        print(f"  {hand_name.capitalize()}: Interpolating frames {gap_start}-{gap_end-1} "
                              f"(gap={gap_length}, from frame {last_valid_idx} to {gap_end}, "
                              f"distance={center_distance:.1f}px < {max_distance:.1f}px)")
                        
                        # Total distance includes the step from last_valid to gap_start
                        # and from gap_end-1 to gap_end
                        total_steps = gap_end - last_valid_idx
                        
                        # Also get raw bboxes for interpolation
                        start_raw_bbox = raw_data[last_valid_idx].get(raw_bbox_key)
                        end_raw_bbox = raw_data[gap_end].get(raw_bbox_key)

                        for j in range(gap_start, gap_end):
                            # Linear interpolation: evenly distribute motion across all steps
                            alpha = (j - last_valid_idx) / total_steps
                            interp_bbox = (1 - alpha) * start_bbox + alpha * end_bbox
                            raw_data[j][bbox_key] = interp_bbox

                            # Interpolate raw_bbox too
                            if start_raw_bbox is not None and end_raw_bbox is not None:
                                raw_data[j][raw_bbox_key] = (1 - alpha) * start_raw_bbox + alpha * end_raw_bbox
                            else:
                                raw_data[j][raw_bbox_key] = None

                            # Create dummy keypoints
                            dummy_keypoints = np.zeros((21, 3))
                            dummy_keypoints[:, 2] = 0.5
                            raw_data[j][keyp_key] = dummy_keypoints
                            raw_data[j][conf_key] = 0.5
                            interpolated_count += 1
                        
                        i = gap_end
                    else:
                        # Bboxes are too far apart - DON'T interpolate, treat as separate instances
                        print(f"  {hand_name.capitalize()}: Skipping interpolation for frames {gap_start}-{gap_end-1} "
                              f"(distance={center_distance:.1f}px > {max_distance:.1f}px - likely different hand instances)")
                        
                        # Move to the next detection without filling the gap
                        i = gap_end
                else:
                    # Hand never reappeared within patience threshold
                    # Apply limited patience (only MAX_PATIENCE_WITHOUT_RETURN frames), then skip the rest
                    last_bbox = raw_data[last_valid_idx][bbox_key]
                    last_raw_bbox = raw_data[last_valid_idx].get(raw_bbox_key)
                    frames_to_fill = min(MAX_PATIENCE_WITHOUT_RETURN, len(raw_data) - gap_start)

                    if frames_to_fill > 0:
                        print(f"  {hand_name.capitalize()}: Applying static patience for frames {gap_start}-{gap_start + frames_to_fill - 1} "
                              f"(hand never reappeared, using last bbox from frame {last_valid_idx})")

                    for j in range(gap_start, gap_start + frames_to_fill):
                        raw_data[j][bbox_key] = last_bbox.copy()
                        raw_data[j][raw_bbox_key] = last_raw_bbox.copy() if last_raw_bbox is not None else None
                        
                        # Create dummy keypoints
                        dummy_keypoints = np.zeros((21, 3))
                        dummy_keypoints[:, 2] = 0.5
                        raw_data[j][keyp_key] = dummy_keypoints
                        raw_data[j][conf_key] = 0.5
                        patience_applied_count += 1
                    
                    # Skip ahead to find where hand reappears (or end of sequence)
                    next_valid_idx = None
                    for j in range(gap_start + frames_to_fill, len(raw_data)):
                        if raw_data[j][bbox_key] is not None:
                            next_valid_idx = j
                            break
                    
                    # Jump to next valid detection or end of sequence
                    i = next_valid_idx if next_valid_idx is not None else len(raw_data)
            else:
                i += 1
    
    if interpolated_count > 0:
        print(f"  Interpolated {interpolated_count} frames where hand reappeared within patience threshold")
    if patience_applied_count > 0:
        print(f"  Applied static patience to {patience_applied_count} frames where hand never reappeared")
    if interpolated_count == 0 and patience_applied_count == 0:
        print(f"  No patience needed")
    
    # Remove spurious short motions (brief appearances after long absence)
    # This runs AFTER patience to filter out hallucinations that patience might have extended
    print("\n" + "-"*80)
    print("Step 4: Removing spurious short motions")
    print("-"*80)
    MIN_MOTION_DURATION = 30  # Motion must last at least this many frames
    MIN_ABSENCE_BEFORE = 30   # Must be absent for this long before
    MIN_ABSENCE_AFTER = 30    # Must be absent for this long after
    
    for hand_name in ['left', 'right']:
        bbox_key = f'{hand_name}_bbox'
        keyp_key = f'{hand_name}_keypoints'
        conf_key = f'{hand_name}_conf'
        
        # Find all continuous segments where hand is present
        segments = []
        start_idx = None
        
        for i, frame_data in enumerate(raw_data):
            if frame_data[bbox_key] is not None:
                if start_idx is None:
                    start_idx = i  # Start of a new segment
            else:
                if start_idx is not None:
                    # End of a segment
                    segments.append((start_idx, i - 1))
                    start_idx = None
        
        # Don't forget the last segment if it extends to the end
        if start_idx is not None:
            segments.append((start_idx, len(raw_data) - 1))
        
        # Check each segment
        removed_segments = []
        for seg_start, seg_end in segments:
            seg_duration = seg_end - seg_start + 1
            
            # Check absence before
            absence_before = seg_start  # frames before this segment
            
            # Check absence after
            absence_after = len(raw_data) - 1 - seg_end  # frames after this segment
            
            # If this is a short motion after long absence and before long absence, remove it
            if (seg_duration < MIN_MOTION_DURATION and 
                absence_before >= MIN_ABSENCE_BEFORE and 
                absence_after >= MIN_ABSENCE_AFTER):
                
                print(f"  {hand_name.capitalize()}: Removing spurious short motion at frames {seg_start}-{seg_end} "
                      f"(duration={seg_duration}, absence_before={absence_before}, absence_after={absence_after})")
                
                # Remove this segment
                for i in range(seg_start, seg_end + 1):
                    raw_data[i][bbox_key] = None
                    raw_data[i][f'{hand_name}_raw_bbox'] = None
                    raw_data[i][keyp_key] = None
                    raw_data[i][conf_key] = 0.0
                
                removed_segments.append((seg_start, seg_end))
        
        if len(removed_segments) > 0:
            print(f"  {hand_name.capitalize()}: Removed {len(removed_segments)} spurious short motion segments")
        else:
            print(f"  {hand_name.capitalize()}: No spurious short motions found")
    
    # raw_data = interpolate_missing_bboxes(raw_data, max_gap=10)
    # # Bbox smoothing disabled - it was causing drift (e.g., frames 326->330)
    # raw_data = smooth_bbox_sequence(raw_data, window_size=5)
    
    # Re-check for overlaps after interpolation (interpolation can create new overlaps)
    print("\n" + "-"*80)
    print("Step 4: Re-checking overlaps after interpolation")
    print("-"*80)
    raw_data = detect_overlapping_bboxes(raw_data, iou_threshold=0.7)
    
    # Restore hands that were removed due to overlap, but ONLY if the "winner" hand was 
    # subsequently invalidated by other checks (handedness consistency).
    # Logic: If LEFT was removed because RIGHT had higher confidence, but then RIGHT itself
    # was removed by handedness check, we should restore LEFT since it was the original detection.
    print("\n" + "-"*80)
    print("Step 5: Restoring hands removed by overlap if winner was invalidated")
    print("-"*80)
    restored_count = 0
    for i, (frame_data, orig) in enumerate(zip(raw_data, original_data)):
        for hand_name in ['left', 'right']:
            bbox_key = f'{hand_name}_bbox'
            keyp_key = f'{hand_name}_keypoints'
            conf_key = f'{hand_name}_conf'
            removal_flag = f'{hand_name}_removed_due_to_overlap_with'
            
            # Check if this hand was removed due to overlap
            if frame_data.get(removal_flag) is not None:
                other_hand = frame_data[removal_flag]  # The hand that "won" the overlap check
                other_bbox_key = f'{other_hand}_bbox'
                other_handedness_flag = f'{other_hand}_removed_by_handedness_check'
                
                # Check if the "winner" hand was subsequently removed by handedness check
                if frame_data.get(other_handedness_flag, False):
                    print(f"  Frame {i}: Restoring {hand_name} hand (winner '{other_hand}' was invalidated)")
                    # Restore original keypoints and confidence
                    if orig[keyp_key] is not None:
                        frame_data[bbox_key] = orig[bbox_key].copy()
                        frame_data[keyp_key] = orig[keyp_key].copy()
                        frame_data[conf_key] = orig[conf_key]
                    else:
                        # Create dummy keypoints if original didn't have them
                        dummy_keypoints = np.zeros((21, 3))
                        dummy_keypoints[:, 2] = 0.5
                        frame_data[keyp_key] = dummy_keypoints
                        frame_data[conf_key] = 0.5
                    # Clear the removal flag since we've restored it
                    frame_data[removal_flag] = None
                    restored_count += 1
    
    raw_data = fix_handedness_swaps_frame_to_frame(raw_data, iou_threshold=0.6)

    if restored_count > 0:
        print(f"  Restored {restored_count} hand instances")
    else:
        print(f"  No hands needed restoration")
    
    # Visualize before/after comparison
    dash_length = 10
    if vis_dir is not None:
        for i, frame_data in enumerate(raw_data):
            img_path = frame_data['img_path']
            img_cv2 = cv2.imread(img_path)
            img_vis = img_cv2.copy()
            
            # Draw original bboxes (dashed)
            orig = original_data[i]
            if orig['left_bbox'] is not None:
                x1, y1, x2, y2 = orig['left_bbox'].astype(int)
                # Draw dashed rectangle

                for j in range(x1, x2, dash_length * 2):
                    cv2.line(img_vis, (j, y1), (min(j + dash_length, x2), y1), (144, 238, 144), 2)
                    cv2.line(img_vis, (j, y2), (min(j + dash_length, x2), y2), (144, 238, 144), 2)
                for j in range(y1, y2, dash_length * 2):
                    cv2.line(img_vis, (x1, j), (x1, min(j + dash_length, y2)), (144, 238, 144), 2)
                    cv2.line(img_vis, (x2, j), (x2, min(j + dash_length, y2)), (144, 238, 144), 2)
                cv2.putText(img_vis, "L_before", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (144, 238, 144), 2)

            if orig['right_bbox'] is not None:
                x1, y1, x2, y2 = orig['right_bbox'].astype(int)
                for j in range(x1, x2, dash_length * 2):
                    cv2.line(img_vis, (j, y1), (min(j + dash_length, x2), y1), (173, 216, 230), 2)
                    cv2.line(img_vis, (j, y2), (min(j + dash_length, x2), y2), (173, 216, 230), 2)
                for j in range(y1, y2, dash_length * 2):
                    cv2.line(img_vis, (x1, j), (x1, min(j + dash_length, y2)), (173, 216, 230), 2)
                    cv2.line(img_vis, (x2, j), (x2, min(j + dash_length, y2)), (173, 216, 230), 2)
                cv2.putText(img_vis, "R_before", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (173, 216, 230), 2)

            # Draw cleaned bboxes (solid) and keypoints
            if frame_data['left_bbox'] is not None:
                x1, y1, x2, y2 = frame_data['left_bbox'].astype(int)
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green for left
                cv2.putText(img_vis, "L_after", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Draw left hand keypoints in green
                if frame_data['left_keypoints'] is not None:
                    for kp in frame_data['left_keypoints']:
                        x, y, conf = kp
                        if conf > 0.1:  # Only draw confident keypoints
                            cv2.circle(img_vis, (int(x), int(y)), 3, (0, 255, 0), -1)  # Green circles

            if frame_data['right_bbox'] is not None:
                x1, y1, x2, y2 = frame_data['right_bbox'].astype(int)
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Blue for right
                cv2.putText(img_vis, "R_after", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Draw right hand keypoints in blue
                if frame_data['right_keypoints'] is not None:
                    for kp in frame_data['right_keypoints']:
                        x, y, conf = kp
                        if conf > 0.1:  # Only draw confident keypoints
                            cv2.circle(img_vis, (int(x), int(y)), 3, (255, 0, 0), -1)  # Blue circles
            
            img_fn = os.path.splitext(os.path.basename(img_path))[0]
            cv2.imwrite(os.path.join(pass2_dir, f'{img_fn}_cleaned.jpg'), img_vis)
    
    # Create video from Pass 2 visualizations
    if vis_dir is not None:
        pass2_dir = os.path.join(vis_dir, 'pass2_cleaned_bboxes')
        video_path = os.path.join(vis_dir, 'pass2_cleaned_bboxes.mp4')
        create_video_from_images(pass2_dir, video_path, fps=fps)
        
        # Plot bbox trajectories to visualize handedness consistency
        plot_bbox_trajectories(raw_data, vis_dir, filename='bbox_trajectories_after_cleaning.png')
    
    print("\n" + "="*80)
    print("PASS 2 Complete: Bbox sequences cleaned")
    print("="*80)
    
    return raw_data


def convert_crop_coords_to_orig_img(bbox, keypoints, crop_size):
    """Convert cropped coordinates to original image coordinates."""
    cx, cy, h = bbox[:, 0], bbox[:, 1], bbox[:, 2]
    keypoints *= h[..., None, None] / crop_size
    keypoints[:,:,0] = (cx - h/2)[..., None] + keypoints[:,:,0]
    keypoints[:,:,1] = (cy - h/2)[..., None] + keypoints[:,:,1]
    return keypoints
