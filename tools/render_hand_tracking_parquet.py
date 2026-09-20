"""Render final joint tracks without rerunning the observation/HMR pipeline."""

import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parquet', required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--depth_dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--camera', choices=['left', 'right'], required=True)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--max_rmse_m', type=float, default=0.03)
    parser.add_argument('--mano_path', default=str(ROOT / '_DATA/data/mano'))
    args = parser.parse_args()
    from hmr_backends.models.mano_wrapper import MANO
    from hmr_backends.utils.renderer import Renderer
    from hmr_backends.utils.parquet_mesh import render_tracking_parquet

    images = sorted(p for p in Path(args.images).iterdir()
                    if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if not images or args.fps <= 0:
        parser.error('Nonempty image directory and positive fps required')
    frames = [{'frame_idx': i, 'img_path': str(path)} for i, path in enumerate(images)]
    mano = MANO(model_path=args.mano_path, use_pca=False).to(args.device).eval()
    config = SimpleNamespace(EXTRA=SimpleNamespace(FOCAL_LENGTH=5000),
                             MODEL=SimpleNamespace(IMAGE_SIZE=256))
    renderer = Renderer(config, faces=mano.faces)
    report = render_tracking_parquet(
        args.parquet, frames, mano, renderer, args.output, depth_dir=args.depth_dir,
        expected_reference_camera=args.camera, fps=args.fps,
        steps=args.steps, max_rmse_m=args.max_rmse_m,
    )
    print(f"Rendered {report['rendered_hands']}/{report['present_hands']} hands")


if __name__ == '__main__':
    main()
