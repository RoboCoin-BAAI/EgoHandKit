from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import cv2
import numpy as np
import pyrender
import trimesh
from tqdm import tqdm

from bbox_utils import create_video_from_images


WORLD_GRID_BACKGROUND_BGR = np.array([180, 180, 180], dtype=np.uint8)
WORLD_GRID_BACKGROUND_RGB = tuple((WORLD_GRID_BACKGROUND_BGR[::-1].astype(np.float32) / 255.0).tolist())
LIGHT_GRAY_BGR = WORLD_GRID_BACKGROUND_BGR
LIGHT_GRAY_RGB = WORLD_GRID_BACKGROUND_RGB
COLOR_GREEN_RGB = (0.2, 0.8, 0.3)
COLOR_BLUE_RGB = (0.2, 0.4, 0.9)
CAMERA_MARKER_RGB = (1.0, 0.0, 0.0)
DEFAULT_WORLD_VIEW_DISTANCE_SCALE = 2.2
DEFAULT_WORLD_VIEW_FOCAL_SCALE = 0.55
DEFAULT_CAMERA_MARKER_RADIUS_SCALE = 0.03


@dataclass(frozen=True)
class FixedWorldView:
    name: str
    w2c: np.ndarray
    intrinsic: np.ndarray
    eye: np.ndarray
    target: np.ndarray


def compute_world_bounds(omega_world_results: dict, min_radius: float = 0.05) -> tuple[np.ndarray, float]:
    points = []
    for frame in omega_world_results.values():
        for vertices in frame.get("world_vertices", []):
            vertices = np.asarray(vertices, dtype=np.float32)
            if vertices.size == 0:
                continue
            vertices = vertices.reshape(-1, 3)
            vertices = vertices[np.isfinite(vertices).all(axis=1)]
            if len(vertices) > 0:
                points.append(vertices)
        c2w = frame.get("omega_c2w")
        if c2w is not None:
            c2w = np.asarray(c2w, dtype=np.float32)
            if c2w.shape == (4, 4) and np.isfinite(c2w[:3, 3]).all():
                points.append(c2w[:3, 3][None])
    if not points:
        return np.zeros(3, dtype=np.float32), 1.0
    all_points = np.concatenate(points, axis=0)
    p_min = all_points.min(axis=0)
    p_max = all_points.max(axis=0)
    center = ((p_min + p_max) * 0.5).astype(np.float32)
    radius = float(np.linalg.norm(p_max - center))
    return center, max(radius, float(min_radius))


def transform_world_vertices_to_camera(vertices_world: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices_world, dtype=np.float32).reshape(-1, 3)
    pose = np.asarray(w2c, dtype=np.float32)
    return (vertices @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)


def build_camera_marker_world(
    c2w: np.ndarray,
    radius: float,
    height: float,
    color_rgb: tuple[float, float, float] = CAMERA_MARKER_RGB,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c2w = np.asarray(c2w, dtype=np.float32)
    if c2w.shape != (4, 4):
        raise ValueError(f"c2w must have shape (4, 4), got {c2w.shape}")
    radius = float(radius)
    height = float(height)
    vertices_cam = np.array(
        [
            [-radius, -radius, 0.0],
            [radius, -radius, 0.0],
            [radius, radius, 0.0],
            [-radius, radius, 0.0],
            [0.0, 0.0, -height],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [1, 0, 4],
            [2, 1, 4],
            [3, 2, 4],
            [0, 3, 4],
        ],
        dtype=np.int64,
    )
    vertices_world = (vertices_cam @ c2w[:3, :3].T + c2w[:3, 3]).astype(np.float32)
    face_colors = np.array([(*color_rgb, 1.0)] * len(faces), dtype=np.float32)
    return vertices_world, faces, face_colors


def build_fixed_world_views(
    center: np.ndarray,
    radius: float,
    reference_c2w: np.ndarray,
    image_size: tuple[int, int],
) -> list[FixedWorldView]:
    center = np.asarray(center, dtype=np.float32)
    reference_c2w = np.asarray(reference_c2w, dtype=np.float32)
    img_h, img_w = image_size
    radius = max(float(radius), 1e-4)
    distance = radius * DEFAULT_WORLD_VIEW_DISTANCE_SCALE
    focal = DEFAULT_WORLD_VIEW_FOCAL_SCALE * float(min(img_w, img_h)) * distance / radius
    intrinsic = np.array(
        [
            [focal, 0.0, img_w / 2.0],
            [0.0, focal, img_h / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    scene_right = _normalize(reference_c2w[:3, 0], "reference camera right")
    scene_down = _normalize(reference_c2w[:3, 1], "reference camera down")
    scene_forward = _normalize(reference_c2w[:3, 2], "reference camera forward")
    scene_up = -scene_down

    view_specs = [
        ("front", center - scene_forward * distance, scene_up),
        ("right", center - scene_right * distance, scene_up),
        ("top", center + scene_up * distance, scene_forward),
    ]
    views = []
    for name, eye, up_hint in view_specs:
        w2c = _look_at_w2c_cv(eye=eye, target=center, up_hint=up_hint)
        views.append(
            FixedWorldView(
                name=name,
                w2c=w2c,
                intrinsic=intrinsic.copy(),
                eye=eye.astype(np.float32),
                target=center.copy(),
            )
        )
    return views


def compose_grid_frame(
    top_left_bgr: np.ndarray | None,
    world_tiles_bgr: Iterable[np.ndarray | None],
    tile_size: tuple[int, int],
    background_bgr: np.ndarray = WORLD_GRID_BACKGROUND_BGR,
) -> np.ndarray:
    img_h, img_w = tile_size
    tiles = [_fit_tile_bgr(top_left_bgr, tile_size, background_bgr)]
    world_tiles = list(world_tiles_bgr)
    for idx in range(3):
        tile = world_tiles[idx] if idx < len(world_tiles) else None
        tiles.append(_fit_tile_bgr(tile, tile_size, background_bgr))
    top = np.concatenate([tiles[0], tiles[1]], axis=1)
    bottom = np.concatenate([tiles[2], tiles[3]], axis=1)
    grid = np.concatenate([top, bottom], axis=0)
    expected_shape = (img_h * 2, img_w * 2, 3)
    if grid.shape != expected_shape:
        raise ValueError(f"Composed grid has shape {grid.shape}, expected {expected_shape}")
    return grid


def render_omega_world_grid_video(
    omega_world_results: dict,
    image_paths: list[str],
    out_dir: str | Path,
    backend_name: str,
    renderer,
    fps: float,
) -> dict:
    out_dir = Path(out_dir)
    grid_dir = out_dir / f"omega_world_grid_{backend_name}"
    grid_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"omega_world_grid_{backend_name}.mp4"
    overlay_dir = out_dir / f"render_{backend_name}"

    first_image = _read_first_image(image_paths)
    img_h, img_w = first_image.shape[:2]
    tile_size = (img_h, img_w)
    center, radius = compute_world_bounds(omega_world_results)
    reference_c2w = _first_reference_c2w(omega_world_results)
    views = build_fixed_world_views(center, radius, reference_c2w, tile_size)
    camera_marker_radius = max(radius * DEFAULT_CAMERA_MARKER_RADIUS_SCALE, 0.02)

    print("\nRendering Omega world fixed-view grid")
    print(f"  Output frames: {grid_dir}")
    print(f"  Output video:  {video_path}")
    print(f"  Views:         {', '.join(view.name for view in views)}")
    for img_path in tqdm(image_paths, desc="Omega world grid"):
        img_path = str(img_path)
        frame = omega_world_results.get(img_path)
        overlay = _read_overlay_or_image(img_path, overlay_dir, tile_size)
        world_tiles = [
            render_world_view_tile_bgr(
                frame,
                view,
                renderer,
                tile_size,
                camera_marker_radius=camera_marker_radius,
            )
            for view in views
        ]
        grid = compose_grid_frame(overlay, world_tiles, tile_size)
        frame_name = Path(img_path).stem
        cv2.imwrite(str(grid_dir / f"{frame_name}.jpg"), grid)

    create_video_from_images(grid_dir, video_path, fps=fps)
    return {
        "frame_dir": grid_dir,
        "video_path": video_path,
        "views": views,
        "center": center,
        "radius": radius,
    }


def render_world_view_tile_bgr(
    frame: dict | None,
    view: FixedWorldView,
    renderer,
    tile_size: tuple[int, int],
    background_bgr: np.ndarray = WORLD_GRID_BACKGROUND_BGR,
    camera_marker_radius: float = 0.03,
) -> np.ndarray:
    img_h, img_w = tile_size
    background = np.full((img_h, img_w, 3), background_bgr, dtype=np.uint8)
    if frame is None:
        return background

    mesh_specs = []
    for hand_idx, vertices_world in enumerate(frame.get("world_vertices", [])):
        vertices_world = np.asarray(vertices_world, dtype=np.float32)
        if vertices_world.size == 0 or not np.isfinite(vertices_world).all():
            continue
        vertices = transform_world_vertices_to_camera(vertices_world, view.w2c)
        if vertices[:, 2].mean() <= 1e-4:
            continue
        right = _frame_hand_is_right(frame, hand_idx)
        faces = renderer.faces if right else renderer.faces_left
        mesh_specs.append(
            {
                "vertices": vertices,
                "faces": faces,
                "vertex_color": COLOR_BLUE_RGB if right else COLOR_GREEN_RGB,
            }
        )

    c2w = frame.get("omega_c2w")
    if c2w is not None:
        marker_world, marker_faces, marker_face_colors = build_camera_marker_world(
            c2w,
            radius=camera_marker_radius,
            height=camera_marker_radius * 2.0,
        )
        marker_cam = transform_world_vertices_to_camera(marker_world, view.w2c)
        if marker_cam[:, 2].mean() > 1e-4:
            mesh_specs.append(
                {
                    "vertices": marker_cam,
                    "faces": marker_faces,
                    "face_colors": marker_face_colors,
                }
            )

    if not mesh_specs:
        return background

    rgba = _render_mesh_specs_rgba(
        mesh_specs,
        renderer,
        tile_size,
        focal_length=float(view.intrinsic[0, 0]),
        background_rgb=WORLD_GRID_BACKGROUND_RGB,
    )
    rgb = _alpha_composite_on_background(rgba, WORLD_GRID_BACKGROUND_RGB)
    return np.clip(rgb[:, :, ::-1] * 255.0, 0, 255).astype(np.uint8)


def _render_mesh_specs_rgba(
    mesh_specs: list[dict],
    renderer,
    tile_size: tuple[int, int],
    focal_length: float,
    background_rgb: tuple[float, float, float],
) -> np.ndarray:
    img_h, img_w = tile_size
    offscreen = pyrender.OffscreenRenderer(
        viewport_width=img_w,
        viewport_height=img_h,
        point_size=1.0,
    )
    scene = pyrender.Scene(
        bg_color=[*background_rgb, 0.0],
        ambient_light=(0.3, 0.3, 0.3),
    )
    opengl_rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    for mesh_idx, spec in enumerate(mesh_specs):
        vertices = np.asarray(spec["vertices"], dtype=np.float32)
        faces = np.asarray(spec["faces"], dtype=np.int64)
        if "face_colors" in spec:
            mesh = trimesh.Trimesh(
                vertices=vertices.copy(),
                faces=faces.copy(),
                face_colors=np.asarray(spec["face_colors"], dtype=np.float32),
                process=False,
            )
        else:
            color = spec.get("vertex_color", (1.0, 1.0, 1.0))
            vertex_colors = np.array([(*color, 1.0)] * len(vertices), dtype=np.float32)
            mesh = trimesh.Trimesh(
                vertices=vertices.copy(),
                faces=faces.copy(),
                vertex_colors=vertex_colors,
                process=False,
            )
        mesh.apply_transform(opengl_rot)
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False), f"mesh_{mesh_idx}")

    camera = pyrender.IntrinsicsCamera(
        fx=focal_length,
        fy=focal_length,
        cx=img_w / 2.0,
        cy=img_h / 2.0,
        zfar=1e12,
    )
    camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
    scene.add_node(camera_node)
    renderer.add_lighting(scene, camera_node)
    color, _ = offscreen.render(scene, flags=pyrender.RenderFlags.RGBA)
    offscreen.delete()
    return color.astype(np.float32) / 255.0


def _alpha_composite_on_background(rgba: np.ndarray, background_rgb: tuple[float, float, float]) -> np.ndarray:
    alpha = rgba[:, :, 3:4]
    background = np.array(background_rgb, dtype=np.float32).reshape(1, 1, 3)
    return rgba[:, :, :3] * alpha + background * (1.0 - alpha)


def _look_at_w2c_cv(eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    forward = _normalize(target - eye, "camera forward")
    right = np.cross(forward, up_hint)
    if np.linalg.norm(right) < 1e-6:
        right = _fallback_right_axis(forward)
    right = _normalize(right, "camera right")
    down = _normalize(np.cross(forward, right), "camera down")
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.stack([right, down, forward], axis=1)
    c2w[:3, 3] = eye
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = c2w[:3, :3].T
    w2c[:3, 3] = -w2c[:3, :3] @ eye
    return w2c


def _fallback_right_axis(forward: np.ndarray) -> np.ndarray:
    for candidate in np.eye(3, dtype=np.float32):
        right = np.cross(forward, candidate)
        if np.linalg.norm(right) >= 1e-6:
            return right
    raise ValueError("Could not construct a camera right axis")


def _normalize(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if norm < 1e-8:
        raise ValueError(f"Cannot normalize near-zero {name}: {value}")
    return (value / norm).astype(np.float32)


def _fit_tile_bgr(
    image_bgr: np.ndarray | None,
    tile_size: tuple[int, int],
    background_bgr: np.ndarray,
) -> np.ndarray:
    img_h, img_w = tile_size
    if image_bgr is None:
        return np.full((img_h, img_w, 3), background_bgr, dtype=np.uint8)
    image = np.asarray(image_bgr)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape[:2] != tile_size:
        image = cv2.resize(image, (img_w, img_h), interpolation=cv2.INTER_AREA)
    return image


def _read_overlay_or_image(img_path: str, overlay_dir: Path, tile_size: tuple[int, int]) -> np.ndarray:
    overlay_path = overlay_dir / f"{Path(img_path).stem}.jpg"
    image = cv2.imread(str(overlay_path)) if overlay_path.exists() else None
    if image is None:
        image = cv2.imread(str(img_path))
    return _fit_tile_bgr(image, tile_size, LIGHT_GRAY_BGR)


def _read_first_image(image_paths: list[str]) -> np.ndarray:
    if not image_paths:
        raise ValueError("render_omega_world_grid_video requires at least one image path")
    image = cv2.imread(str(image_paths[0]))
    if image is None:
        raise ValueError(f"Could not read first image: {image_paths[0]}")
    return image


def _first_reference_c2w(omega_world_results: dict) -> np.ndarray:
    for frame in omega_world_results.values():
        c2w = frame.get("omega_c2w")
        if c2w is not None:
            return np.asarray(c2w, dtype=np.float32)
    return np.eye(4, dtype=np.float32)


def _frame_hand_is_right(frame: dict, hand_idx: int) -> bool:
    mano = frame.get("mano", [])
    if hand_idx < len(mano) and "is_right" in mano[hand_idx]:
        return int(mano[hand_idx]["is_right"]) == 1
    tracked_ids = frame.get("tracked_ids", [])
    if hand_idx < len(tracked_ids):
        return int(tracked_ids[hand_idx]) == 1
    return True
